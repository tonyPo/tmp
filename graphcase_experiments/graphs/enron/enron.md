# Enron

The Enron E-mail dataset consist of the E-mail inbox of 158 Enron employees. Shetty and Adibi, 2004 have added job title to these employees. The 158 Enron employees are marked as "core" enployees.

Agarwal et al extend this work by adding job titles for non core Enron employee that appear in one of the mails in the inboxes or outboxes of the core employees, extending the size to 1518 employees.

## edge attributes
We have used the following edge attributes:

- src (sender)
- dst (receiver)
- length (length of E-mail body in bytes)
- cnt_to
- cnt_cc
- cnt_bcc
- is_to
- is_cc
- is_bcc

## node atributes
- cnt_received_to
- cnt_recevied_cc
- cnt_received_bcc
- cnt_sent
- total_size_received
- total_size_sent
- is_enron
- is_core

## data
The E-mail corpus is taken from (https://www.cs.cmu.edu/~enron/) distributed by the CALO Project
## label core group
functions and labels are taken from: 
"Discovering Organizational Hierarchy through a Corporate Ranking Algorithm: The Enron Case"
by G. Creamer et al.



These labels are available in de SEGK [code(https://github.com/giannisnik/segk/blob/master/datasets/enron/employees.txt).
labels (https://github.com/enrondata/enrondata/blob/master/data/misc/edo_enron-custodians-data.html)

## labels extended
Agarwal et all created an extended labeling of non core employees. These are on a [mongoDB](http://www.cs.columbia.edu/~rambow/enron/index.html) 


## additional resources
(https://github.com/enrondata/enrondata/tree/master/data). 

http://enrondata.org/en/latest/data/custodian-names-and-titles/. 

https://new.pythonforengineers.com/blog/analysing-the-enron-email-corpus/
