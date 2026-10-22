import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
APP=Path(__file__).resolve().parents[1]/'app.py'
class InterfaceTests(unittest.TestCase):
    def test_navigation(self):
        app=AppTest.from_file(str(APP),default_timeout=30).run()
        self.assertFalse(app.exception)
        for page in ['Text analysis','Topics & keywords','Trends & alerts','Model evaluation','Data overview']:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception,msg=page)
    def test_load_benchmark(self):
        app=AppTest.from_file(str(APP),default_timeout=30).run()
        app.selectbox[0].set_value('Supplied Dreaddit benchmark').run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertGreater(len(app.session_state['data']),3000)
if __name__=='__main__': unittest.main()
