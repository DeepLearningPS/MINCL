ddi_minerals.tsv
This benchmark contains drug drug interactions and the effect they have on minerals
Format
<drug> <mineral_effect> <drug> #药物与药物作用，对疾病的影响
Undirected

ddi_efficacy.tsv
This benchmark contains drug drug interactions and the effect they have on theraputic efficacy #治疗功效，增加或减弱药物的影响
Format
<drug> <efficact_effect> <drug>
Undirected

dpi_fda.tsv
This benchmark contains drug protein interactions for FDA approved drugs #该基准包含FDA批准的药物的药物蛋白相互作用，即药物与蛋白之间的作用，药物对蛋白的影响，即药物对蛋白表达的影响
Format
<drug> DPI <protein>
Directed

dep_fda_exp.tsv
This benchmark contains drug protein interaction and the effect they have on protein expression for FDA approved drugs #该基准包含药物蛋白相互作用以及它们对FDA批准的药物蛋白表达的影响，比上面的数据多了关系中：上调与下调，即药物对蛋白的好与坏的影响
Format
<drug> inc_expr|dec_expr <protein>
Directed

phosphorylation.tsv
This benchmark contains protein protein phosphorylation interactions #该基准包含蛋白质之间的磷酸化相互作用
Format
<kinase_protein> phosphorylates <substrate_protein> <substrate_site>

蛋白激酶， 使磷酸化，基质蛋白 培养基点位；<substrate_protein> <substrate_site> 这里2个人选一个即可？把最后两列合在一起

Directed

总之分药物药物、药物与蛋白、蛋白与蛋白