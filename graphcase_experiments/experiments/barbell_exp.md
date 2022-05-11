## experiment setup

- create a directed semi complete barbell with "bell" size of 10 and path length between the bells of 9.
- perform hyper parameter grid search on it with 200 trails with Optuna.
- select the best performing hyperparameter set.
- train the model with the best hyperparameter set with 10.000 epochs.
- calculcate the embedding for all nodes in the  graph and color then by label.


## results

<img src="https://github.com/tonyPo/graphcase_experiments/blob/main/graphcase_experiments/graphs/barbellgraphs/images/graphCase_embed_barbell.png?raw=true" alt="Barbell embeding GraphCASE" width="800"/>

| id     | embed1 | embed2 | label |	_	| id     | embed1 | embed2 | label |
|-------:|---------:|---------:|----| -	|-------:|---------:|---------:|----|
|    0.0 | 0.540470 | 0.155461 | b1 |		|   15.0 | 0.419770 | 0.290855 | b2 |
|    1.0 | 0.540470 | 0.155461 | b1 |		|   16.0 | 0.419770 | 0.290855 | b2 |
|    2.0 | 0.540470 | 0.155461 | b1 |		|   17.0 | 0.419770 | 0.290855 | b2 |
|    3.0 | 0.540470 | 0.155461 | b1 |		|   18.0 | 0.419770 | 0.290855 | b2 |
|    4.0 | 0.540470 | 0.155461 | b1 |		|   19.0 | 0.321977 | 0.406830 | b3 |
|    5.0 | 0.419770 | 0.290855 | b2 |		|   24.0 | 0.043993 | 0.769308 | b4 |
|    6.0 | 0.419770 | 0.290855 | b2 |		|   23.0 | 0.045745 | 0.772503 | b5 |
|    7.0 | 0.419770 | 0.290855 | b2 |		|   25.0 | 0.045745 | 0.772503 | b5 |
|    8.0 | 0.419770 | 0.290855 | b2 |		|   22.0 | 0.043993 | 0.769308 | b6 |
|    9.0 | 0.321977 | 0.406830 | b3 |		|   26.0 | 0.043993 | 0.769308 | b6 |
|   10.0 | 0.540470 | 0.155461 | b1 |		|   21.0 | 0.051959 | 0.755645 | b7 |
|   11.0 | 0.540470 | 0.155461 | b1 |		|   27.0 | 0.051959 | 0.755645 | b7 |
|   12.0 | 0.540470 | 0.155461 | b1 |		|   20.0 | 0.197486 | 0.583612 | b8 |
|   13.0 | 0.540470 | 0.155461 | b1 |		|   28.0 | 0.197486 | 0.583612 | b8 |
|   14.0 | 0.540470 | 0.155461 | b1 |	  |.       |.         |.          |.  |

Note that:
- all nodes with the same label are plotted to the same embedding.
- that the nodes in the path (label 4,5,6,7) are plotted close together 
- that the nodes of the bell (label 1 + 2) are plotted in the opposite corner of the path nodes.
- the two connecting roles (label 3 + label 8) are plotted in the middle

Note that the embedding is trained 2 hops deep, hence the relative big difference between node label 8 and 7.

## training results

<img src="https://github.com/tonyPo/graphcase_experiments/blob/main/graphcase_experiments/graphs/barbellgraphs/images/graphCASE_training_barbell.png?raw=true" alt="Barbell training GraphCASE" width="400"/>
