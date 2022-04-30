#%%
file = '/Users/tonpoppe/Downloads/maildir/keiser-k/inbox/2.'
# %%

from email.parser import BytesParser
from email import policy
with open(file, 'rb') as fp:
    name = fp.name  # Get file name
    msg = BytesParser(policy=policy.default).parse(fp)
# %%
msg.get_all('to')
# %%
msg.get_all('from')
# %%
