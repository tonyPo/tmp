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
- total_received_to
- total_recevied_cc
- total_received_bcc

## label core group
function category of the core group.
These labels are available in de SEGK code.

## labels extended
Agarwal et all created an extended labeling of non core employees. These are on a [mongoDB](http://www.cs.columbia.edu/~rambow/enron/index.html) 


## additional resources
(https://github.com/enrondata/enrondata/tree/master/data)
