import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lifemate.retrieval import CatalogSearch, terms


class RetrievalTests(unittest.TestCase):
    def test_normalization_and_schema_dependency(self):
        self.assertEqual(terms('Đổi lịch'),terms('doi lich'))
        tools=[{'name':'a.change','kind':'write','description':'Dời lịch sự kiện',
                'parameters':{'properties':{'id':{'x-ref':'a.events','description':'lấy từ a.list'}}}},
               {'name':'a.list','kind':'read','description':'Danh sách sự kiện','parameters':{}},
               {'name':'b.read','kind':'read','description':'Giá xăng','parameters':{}},
               {'name':'sys.ask_missing_info','kind':'meta','parameters':{}}]
        search=CatalogSearch(tools,{'apps':[]})
        chosen,meta=search.select(['Dời lịch sự kiện'],top_k=1)
        self.assertEqual({t['name'] for t in chosen},{'a.change','a.list','sys.ask_missing_info'})
        self.assertEqual(meta['seed_names'],['a.change'])
        self.assertFalse(meta['fallback_full'])

    def test_empty_query_fallback_and_determinism(self):
        tools=[{'name':'a.read','parameters':{},'description':'abc'}]
        search=CatalogSearch(tools,{'apps':[]})
        self.assertEqual(search.select([''])[0],tools)
        self.assertTrue(search.select([''])[1]['fallback_full'])
        self.assertEqual(search.select(['abc']),search.select(['abc']))
        with self.assertRaises(ValueError):search.select(['abc'],0)


if __name__=='__main__':unittest.main()
