import os, email, re, dateutil
from dateutil.parser import parse as parsedate
from collections import defaultdict
from email.parser import BytesParser
from email import policy
import pandas as pd

STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])



class EmailInfo(object):
    refs_pat = '<?(?P<ref>.+)>?'
    refs_prog = re.compile(refs_pat)
    contacts_pat = '(([\"\']?(?P<realname>\w[\w\ ]*)[\"\']?)?\s+)?<?(?P<email>[\w.]+@([\w_]+.)+[\w_]+)>?'
    contacts_prog = re.compile(contacts_pat)

    def __init__(self, fname, folder):
        self.folder = folder
        with open(fname, 'rb') as fp:
            msg = BytesParser(policy=policy.default).parse(fp)
        self.load(msg)

    def to_dict(self):
        return vars(self)

    def load(self, msg):
        self.message_id = msg.get('message-id')
        self.email_size = len(msg.get_content() or '')
        self.from_address = msg.get_all('from')  # need to check if E-mails can be sent with multiple froms.
        if msg.get_all('to') is None:
            self.to_address = []
        else:
            self.to_address = [a for n,a in email.utils.getaddresses(msg.get_all('to'))]
        self.cnt_to = len(self.to_address)

        if msg.get_all('cc') is None:
            self.cc_address = []
        else:
            self.cc_address = [a for n,a in email.utils.getaddresses(msg.get_all('cc'))]
        self.cnt_cc = len(self.cc_address)

        if msg.get_all('bcc') is None:
            self.bcc_address = []
        else:
            self.bcc_address = [a for n,a in email.utils.getaddresses(msg.get_all('bcc'))]
        self.cnt_bcc = len(self.bcc_address)

        self.content_type = msg.get_content_type()
        self.x_origin = msg.get('X-Origin')
        self.x_folder = msg.get('X-Folder')

     
  



class EmailWalker(object):

    def __init__(self, root):
        self.root = root
        self.curdir = None
        self.skipped = 0
        self.parsed = 0

    def parse_mails(self, verbose=False):
        pdf = None
        for root, _, files in os.walk(self.root):
            folder = os.path.relpath(root, self.root)
            print('root')
            for fname in files:
                try:
                    if verbose:
                        print(f"Parsing {root}:{fname}")
                    msg = EmailInfo(os.path.join(root, fname), folder).to_dict()
                    if pdf is None:
                        pdf = pd.DataFrame.from_dict(msg, orient='index').transpose()
                    else:
                        tmp = pd.DataFrame.from_dict(msg, orient='index').transpose()
                        pdf = pd.concat([pdf, tmp], axis = 0, ignore_index=True)
                    self.parsed += 1
                except Exception as e:
                    self.skipped += 1
                    pass
        print(f'{self.parsed} E-mails parsed and {self.skipped} files skipped')
        return pdf




        


