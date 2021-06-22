from anytree import Node, NodeMixin, RenderTree, cachedsearch, LevelOrderGroupIter
from anytree.importer import DictImporter
from anytree.exporter import DictExporter
import json
from tqdm import tqdm

# parent node 都是 category node
# leaf node 都是 book node

class CategoryNode(NodeMixin):
    def __init__(self, name: str, cdc_code: str, parent=None, children=None):
        super(CategoryNode, self).__init__()
        self.type = 'category'
        self.name = name
        self.cdc_code = cdc_code
        self.parent = parent
        self.children = children if children else []

    def __repr__(self):
        return str(self.name)


class BookNode(NodeMixin):
    def __init__(self, book_idx: str, cdc_code: str, title: str, 
                parent: NodeMixin, 
                create_date=None,
                **kwargs):
        super(BookNode, self).__init__()
        self.type = 'book'
        self.book_idx = book_idx
        self.cdc_code = cdc_code
        self.create_date = create_date
        self.title = title
        self.parent = parent
        self.__dict__.update(kwargs)
    
    def __repr__(self):
        return str(self.title)


class CDCTree:
    def __init__(self):
        self.root = CategoryNode('root', '-1')
        self.bookid2node = {}
        self.cdc2leaf = {}

    def save(self, file: str):
        exporter = DictExporter()
        tree = exporter.export(self.root)
        output = {'tree': tree}
        with open(file, 'w') as f:
            f.write(json.dumps(output, ensure_ascii=False, indent=2))

    @staticmethod
    def from_built(file:str):
        importer = DictImporter()
        tree = CDCTree()
        with open(file) as f:
            d = json.load(f)
        root = importer.import_(d['tree'])
        tree.root = root
        for book in tqdm(tree.root.leaves):
            tree.bookid2node[book.book_idx] = book
            tree.cdc2leaf[book.cdc_code] = book.parent
        return tree

    def add_book(self,  cdc_code: str, 
                book_idx: str, 
                title: str, 
                create_date: str=None, 
                publisher:str=None,
                **kwargs):
        if len(cdc_code) != 3:
            raise ValueError('cdc_code length must be 3.')
        
        if book_idx in self.bookid2node.keys():
            return self.bookid2node[book_idx]

        if cdc_code in self.cdc2leaf.keys():
            cur = self.cdc2leaf[cdc_code]
        else:
            prev = self.root
            for idx, code in enumerate(cdc_code):
                cur = None
                for tmp in prev.children:
                    if tmp.cdc_code == cdc_code[:idx + 1]:
                        cur = tmp
                        break
                if not cur:
                    cur = CategoryNode(cdc_code[:idx + 1],
                                cdc_code=cdc_code[:idx + 1],
                                parent=prev)
                prev = cur
            self.cdc2leaf[cdc_code] = cur
        book_node = BookNode(book_idx, parent=cur,
                        title=title,
                        times=0,
                        cdc_code=cdc_code,
                        create_date=create_date,
                        publisher=publisher
                        , **kwargs)
        self.bookid2node[book_idx] = book_node
        return book_node
    
    def update(self,  cdc_code: str, 
                book_idx: str, 
                title: str=None, 
                create_date: str=None, 
                publisher: str= None,
                **kwargs):
        existed_book = self.add_book(cdc_code, book_idx, title=title, create_date=create_date, publisher=publisher, **kwargs)
        existed_book.times += 1
        return existed_book
