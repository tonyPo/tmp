#%%
file = '/Users/tonpoppe/Downloads/maildir/keiser-k/inbox/2.'
# %%

from email.parser import BytesParser
from email import policy
with open(file, 'rb') as fp:
    name = fp.name  # Get file name
    msg = BytesParser(policy=policy.default).parse(fp)

    msg.get('CC')
# %%
msg.get_all('to')
# %%
msg.get_all('from')
# %%
import os
os.getcwd()
# %%

from graphcase_experiments.graphs.enron.email_util import EmailInfo, EmailWalker
file = '/Users/tonpoppe/Downloads/maildir/keiser-k/inbox/2.'
res = EmailInfo(file, "test")
msg= res.to_dict()

# %%
from graphcase_experiments.graphs.enron.email_util import EmailInfo, EmailWalker
# root = '/Users/tonpoppe/Downloads/maildir/keiser-k/inbox/'
root = '/Users/tonpoppe/Downloads/maildir/king-j/inbox/'
# root = '/Users/tonpoppe/Downloads/maildir/king-j/'
root = '/Users/tonpoppe/Downloads/testenron/'
emailWalker = EmailWalker(root)
pdf = emailWalker.parse_mails(verbose=False)
pdf.to_pickle('/Users/tonpoppe/Downloads/enron_parsed')
# %%


