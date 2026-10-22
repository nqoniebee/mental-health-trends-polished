"""Run with: python -m streamlit run app.py"""
import base64
import hashlib
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from src.data import ROOT, normalize_posts
from src import classifier, models
from src.trends import aggregate

st.set_page_config(page_title='Mental Health Trends',page_icon='🌿',layout='wide')
background=ROOT/'assets/background.png'
if background.exists():
    encoded=base64.b64encode(background.read_bytes()).decode()
    st.markdown(f'''<style>
    .stApp {{background:linear-gradient(rgba(240,248,247,.86),rgba(240,248,247,.92)),url("data:image/png;base64,{encoded}") center/cover fixed;}}
    [data-testid="stSidebar"] {{background:#153e47;}}
    [data-testid="stSidebar"] * {{color:#f3faf9;}}
    [data-testid="stSidebar"] [data-baseweb="select"] * {{color:#153e47;}}
    [data-testid="stMetric"] {{background:rgba(255,255,255,.90);border:1px solid #d5e7e3;border-radius:16px;padding:16px;}}
    .block-container {{padding-top:2.2rem;max-width:1400px;}}
    h1,h2,h3 {{letter-spacing:-.025em;}}
    </style>''',unsafe_allow_html=True)

st.sidebar.title('🌿 Mind • Trends')
st.sidebar.caption('Social text · Research dashboard')
page=st.sidebar.radio('Workspace',['Data overview','Text analysis','Topics & keywords','Trends & alerts','Model evaluation'])
st.sidebar.divider()
st.sidebar.caption('Aggregate language signals for research. Not a clinical diagnosis or a measure of mental-health prevalence.')
st.title('Early Detection of Mental Health Trends')
st.caption('Explore language. Understand patterns. Review unusual changes.')

@st.cache_resource
def get_bundle():
    return classifier.load()


def failure(exc):
    st.error('This operation could not finish. Your other pages remain available.')
    st.info('Complete SETUP_WINDOWS.bat first. The first setup needs internet to download the pretrained models. If setup already completed, review the error below.')
    with st.expander('Technical details'):
        st.code(f'{type(exc).__name__}: {exc}')


def data_download(frame,name,label='Download CSV'):
    st.download_button(label,frame.to_csv(index=False).encode('utf-8'),name,'text/csv')


def active_data():
    if 'data' not in st.session_state:
        raw=pd.read_csv(ROOT/'data/raw/sample_reddit_mental_health.csv')
        frame,audit=normalize_posts(raw)
        frame['source']='Synthetic demonstration (supplied sample)'
        st.session_state.data=frame; st.session_state.audit=audit
        st.session_state.source='Synthetic demonstration: repeated texts are removed; only 28 unique posts remain.'
    return st.session_state.data


def clear_results():
    for key in ['analysis','topic_result','keywords','topic_quality']:
        st.session_state.pop(key,None)

frame=active_data()

if page=='Data overview':
    st.subheader('1 · Choose your data')
    st.write('Start with the included sample, load the supplied Dreaddit benchmark, or upload an English-language research CSV.')
    choice=st.selectbox('Data source',['Synthetic demonstration','Supplied Dreaddit benchmark','Upload CSV'])
    upload=st.file_uploader('CSV with text and date columns',type=['csv']) if choice=='Upload CSV' else None
    english=st.checkbox('Apply best-effort English-language filtering',value=False)
    if st.button('Load and clean data',type='primary'):
        try:
            if choice=='Upload CSV':
                if upload is None: raise ValueError('Choose a CSV file first.')
                raw=pd.read_csv(upload)
            elif choice=='Supplied Dreaddit benchmark':
                raw=pd.concat([pd.read_csv(ROOT/'data/raw/dreaddit-train.csv'),pd.read_csv(ROOT/'data/raw/dreaddit-test.csv')],ignore_index=True)
            else: raw=pd.read_csv(ROOT/'data/raw/sample_reddit_mental_health.csv')
            cleaned,audit=normalize_posts(raw,english)
            if cleaned.empty: raise ValueError('No usable posts remain. Check text, dates and language settings.')
            cleaned['source']=choice
            st.session_state.data=cleaned;st.session_state.audit=audit
            st.session_state.source=choice
            clear_results();frame=cleaned
            st.success('Data ready. Continue to Text analysis.')
        except Exception as exc: st.error(str(exc))
    st.info(st.session_state.source)
    a,b,c=st.columns(3)
    a.metric('Usable posts',f'{len(frame):,}'); b.metric('Communities',frame.subreddit.nunique()); c.metric('Repeated rows removed',st.session_state.audit['duplicate_text_rows_removed'])
    with st.expander('Cleaning summary and date range',expanded=True):
        st.json(st.session_state.audit)
        st.write(f'{frame.date.min()} to {frame.date.max()}')
        st.caption('URLs and user mentions are removed. This is not guaranteed anonymisation; review free text before sharing. Dates are UTC. Duplicate text is retained only once across the dataset.')
    st.plotly_chart(px.histogram(frame.assign(words=frame.text.str.split().str.len()),x='words',title='Post length distribution'),width='stretch')
    if st.checkbox('Show cleaned text preview'):
        st.dataframe(frame[['text','date','subreddit']].head(30),hide_index=True)
    data_download(frame,'cleaned_posts.csv')

elif page=='Text analysis':
    st.subheader('2 · Analyse language')
    st.info(st.session_state.source)
    mode=st.radio('Analysis mode',['Dataset','One text example'],horizontal=True)
    if mode=='One text example':
        text=st.text_area('English text',placeholder='Paste a short example to inspect the language models…',height=150,max_chars=10000)
        if st.button('Analyse example',type='primary'):
            try:
                cleaned=models.clean_text(text)
                if len(cleaned)<=5: raise ValueError('Enter meaningful text longer than five characters.')
                with st.spinner('Analysing language…'):
                    result=models.annotate(pd.DataFrame({'text':[cleaned]}),get_bundle())
                a,b,c=st.columns(3)
                a.metric('Sentiment',result.sentiment_label.iloc[0].title());b.metric('Emotion',result.emotion_label.iloc[0].title());c.metric('Stress-language probability',f'{result.stress_probability.iloc[0]:.1%}')
                st.caption('This is a text-classification example, not an assessment of the writer. Probability is a model output, not clinical risk.')
            except Exception as exc: failure(exc)
    else:
        communities=st.multiselect('Communities',sorted(frame.subreddit.unique()),default=sorted(frame.subreddit.unique()))
        date_range=st.date_input('Date range',value=(frame.date.min().date(),frame.date.max().date()))
        selected=frame[frame.subreddit.isin(communities)].copy()
        if len(date_range)==2:
            selected=selected[selected.date.dt.date.between(date_range[0],date_range[1])]
        limit=st.number_input('Maximum posts to analyse',min_value=1,max_value=3000,value=600,step=50)
        if st.button('Run analysis',type='primary'):
            try:
                if selected.empty: raise ValueError('No posts match your filters.')
                chosen=selected.sample(n=min(int(limit),len(selected)),random_state=0).sort_values('date')
                with st.spinner('Running sentiment, emotion and stress models…'):
                    st.session_state.analysis=models.annotate(chosen,get_bundle())
                st.session_state.analysis_scope=f'{len(chosen):,} of {len(selected):,} filtered posts; fixed random seed 0.'
                st.success('Analysis complete. Trends & alerts uses these same results.')
            except Exception as exc: failure(exc)
        if 'analysis' in st.session_state:
            result=st.session_state.analysis
            st.caption(st.session_state.analysis_scope+' Results reflect the last Run analysis action.')
            a,b,c=st.columns(3)
            a.metric('Analysed posts',len(result));b.metric('Negative sentiment',f'{result.sentiment_label.eq("negative").mean():.1%}');c.metric('Stress-language share',f'{result.stress_label.mean():.1%}')
            a,b=st.columns(2)
            for column,label,container in [('sentiment_label','Sentiment',a),('emotion_label','Emotion',b)]:
                counts=result[column].value_counts().rename_axis(label).reset_index(name='Posts')
                container.plotly_chart(px.bar(counts,x=label,y='Posts',color=label),width='stretch')
            if st.checkbox('Show analysed text'): st.dataframe(result,hide_index=True)
            data_download(result,'analysed_posts.csv')

elif page=='Topics & keywords':
    st.subheader('3 · Discover themes and changing words')
    limit=st.number_input('Maximum posts for topics',min_value=10,max_value=3000,value=800,step=50)
    minimum=st.slider('Minimum topic size',2,50,5)
    selected=frame.sample(min(int(limit),len(frame)),random_state=0).sort_values('date')
    st.caption(f'Using {len(selected)} unique posts from the current dataset. Topic −1 means unassigned text.')
    if st.button('Discover topics',type='primary'):
        try:
            with st.spinner('Fitting BERTopic…'):
                model,assigned=models.topics(selected,minimum)
                summary=model.get_topic_info()
                weekly=assigned.assign(period=assigned.date.dt.tz_localize(None).dt.to_period('W').dt.start_time).groupby(['period','topic']).size().reset_index(name='posts')
                st.session_state.topic_result=(summary,weekly,assigned[['id','topic']])
                from sklearn.metrics import silhouette_score
                keep=assigned.topic.ge(0)
                labels=assigned.loc[keep,'topic']
                quality=None
                if 1<labels.nunique()<len(labels):
                    vectors=models.embedder().encode(assigned.loc[keep,'text'].tolist(),show_progress_bar=False)
                    quality=float(silhouette_score(vectors,labels,metric='cosine'))
                st.session_state.topic_quality=quality
        except Exception as exc: failure(exc)
    if 'topic_result' in st.session_state:
        summary,weekly,assigned=st.session_state.topic_result
        # BERTopic representative documents are omitted from the default table.
        st.dataframe(summary.drop(columns=['Representative_Docs'],errors='ignore'),hide_index=True)
        if st.session_state.topic_quality is not None: st.metric('Semantic silhouette (excluding outliers)',f'{st.session_state.topic_quality:.3f}')
        else: st.caption('Topic separation cannot be scored without at least two non-outlier clusters.')
        if summary.Topic.eq(-1).all(): st.info('No stable topics found. Try more varied data or a smaller minimum topic size.')
        st.plotly_chart(px.line(weekly,x='period',y='posts',color=weekly.topic.astype(str),markers=True,title='Topics by week'),width='stretch')
        data_download(assigned,'topic_assignments.csv')
    st.divider()
    if st.button('Extract KeyBERT keywords'):
        try:
            with st.spinner('Finding representative phrases…'):
                st.session_state.keywords=models.keywords(tuple(selected.head(500).text))
        except Exception as exc: failure(exc)
    if 'keywords' in st.session_state:
        st.dataframe(st.session_state.keywords,hide_index=True)
        data_download(st.session_state.keywords,'keywords.csv')
    cutoff=st.date_input('Compare vocabulary before and from this date',value=frame.date.median().date())
    if st.button('Compare TF-IDF vocabulary'):
        try:
            change=models.tfidf_change(frame.loc[frame.date.dt.date<cutoff,'text'].tolist(),frame.loc[frame.date.dt.date>=cutoff,'text'].tolist())
            a,b=st.columns(2);a.write('Rising terms');a.dataframe(change.head(15),hide_index=True);b.write('Falling terms');b.dataframe(change.tail(15).sort_values('change'),hide_index=True)
            data_download(change,'vocabulary_changes.csv')
        except ValueError as exc: st.warning(str(exc))
    if st.button('Create word cloud'):
        try:
            from wordcloud import WordCloud
            cloud=WordCloud(width=1200,height=500,background_color='white',colormap='viridis').generate(' '.join(selected.text))
            st.image(cloud.to_array())
        except ValueError as exc: st.warning(str(exc))

elif page=='Trends & alerts':
    st.subheader('4 · Monitor aggregate changes')
    if 'analysis' not in st.session_state:
        st.info('Run dataset analysis in Text analysis first.')
    else:
        st.caption(st.session_state.analysis_scope)
        a,b,c,d=st.columns(4)
        frequency=a.selectbox('Time interval',['Daily','Weekly','Hourly','Monthly','Yearly'])
        signal=b.selectbox('Signal',['Negative sentiment','Stress-language share'])
        window=c.number_input('Previous periods',min_value=2,max_value=60,value=7)
        minimum=d.number_input('Minimum posts per period',min_value=1,max_value=100,value=3)
        threshold=st.slider('Absolute z-score threshold',1.0,4.0,2.0,0.1)
        result=aggregate(st.session_state.analysis,frequency,'negative' if signal=='Negative sentiment' else 'stress',int(window),int(minimum),threshold)
        st.plotly_chart(px.line(result,x='date',y=['share','baseline'],markers=True,title=signal+' and prior-period baseline'),width='stretch')
        flags=result[result.alert]
        st.metric('Flagged periods',len(flags))
        st.dataframe(result[['date','posts','share','baseline','z_score','status']],hide_index=True)
        st.caption('Share ranges from 0 to 1. The current period is excluded from its baseline. Empty or small periods are gaps, not zero distress. A zero-variance baseline with a change gives an infinite z-score. Flags indicate unusual language changes, not diagnoses or forecasts.')
        if result.baseline.notna().sum()==0:
            st.info('Not enough consecutive periods with sufficient posts. Choose a wider interval, analyse more data, or adjust the controls. The deduplicated synthetic sample is too small for reliable trends.')
        if frequency=='Hourly': st.caption('Hourly analysis needs genuine timestamps; date-only inputs are placed at midnight.')
        data_download(result,'trend_alerts.csv')

else:
    st.subheader('5 · Evaluate the model')
    st.write('Frozen all-MiniLM-L6-v2 sentence embeddings → balanced logistic regression → stress/non-stress language.')
    try:
        bundle=get_bundle(); metrics=bundle['metrics']
        cols=st.columns(5)
        for col,key in zip(cols,['accuracy','precision','recall','f1','roc_auc']): col.metric(key.replace('_',' ').title(),f'{metrics[key]:.3f}')
        st.metric('Majority-class baseline accuracy',f'{metrics["baseline_accuracy"]:.3f}')
        matrix=pd.DataFrame(metrics['confusion_matrix'],index=['Actual non-stress','Actual stress'],columns=['Predicted non-stress','Predicted stress'])
        st.plotly_chart(px.imshow(matrix,text_auto=True,color_continuous_scale='Teal',title='Held-out confusion matrix'),width='stretch')
        path=ROOT/'reports/test_predictions.csv'
        if path.exists():
            from sklearn.metrics import roc_curve,precision_recall_curve
            predictions=pd.read_csv(path)
            fpr,tpr,_=roc_curve(predictions.label,predictions.probability)
            precision,recall,_=precision_recall_curve(predictions.label,predictions.probability)
            a,b=st.columns(2)
            a.plotly_chart(px.line(x=fpr,y=tpr,labels={'x':'False-positive rate','y':'True-positive rate'},title='ROC curve'),width='stretch')
            b.plotly_chart(px.line(x=recall,y=precision,labels={'x':'Recall','y':'Precision'},title='Precision–recall curve'),width='stretch')
            st.write('Misclassification count:',int(predictions.label.ne(predictions.prediction).sum()))
            data_download(predictions,'held_out_predictions.csv')
        with st.expander('Training provenance and cleaning audit'): st.json(bundle['metadata'])
        st.caption('Measured benchmark performance is not clinical validation or evidence of Zimbabwean population accuracy. Chapter 4’s sample-data 100% figures are not substituted here.')
    except Exception as exc: failure(exc)
    comparison=ROOT/'reports/tfidf_comparison.csv'
    if comparison.exists():
        with st.expander('Supplementary Chapter 4-style TF-IDF comparison'):
            st.caption('Separate experiment on the cleaned benchmark. The deployed classifier above uses MiniLM embeddings.')
            st.dataframe(pd.read_csv(comparison),hide_index=True)
    with st.expander('Check pretrained sentiment/emotion against human labels'):
        st.write('Upload a small CSV with text, sentiment_gold and/or emotion_gold. Only supplied human labels are used for agreement; predictions are never treated as ground truth.')
        validation=st.file_uploader('Human-labelled validation CSV',type=['csv'],key='validation')
        if st.button('Measure agreement'):
            try:
                from sklearn.metrics import accuracy_score,cohen_kappa_score
                if validation is None: raise ValueError('Choose a validation CSV first.')
                gold=pd.read_csv(validation)
                if 'text' not in gold: raise ValueError('Missing text column.')
                if len(gold)>300: raise ValueError('Use at most 300 rows for a manual spot-check.')
                checked=False
                for kind in ('sentiment','emotion'):
                    column=kind+'_gold'
                    if column in gold:
                        sub=gold.dropna(subset=['text',column]).copy()
                        if sub.empty: continue
                        y=sub[column].str.strip().str.lower()
                        pred=models.classify(sub.text.map(models.clean_text).tolist(),kind)[kind+'_label']
                        st.write(kind.title(),{'rows':len(y),'accuracy':float(accuracy_score(y,pred)),'kappa':float(cohen_kappa_score(y,pred))})
                        checked=True
                if not checked: raise ValueError('No usable human-label rows found.')
            except Exception as exc: failure(exc)
