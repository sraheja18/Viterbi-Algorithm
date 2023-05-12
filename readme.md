There are three files count_freqs_baseline.py,count_freqs.py and count_freqs_laplace.py
count_freqs_baseline is for the unigram and Trigram HMM where rare words are categorized only in one class _RARE_.
Whereas count_freqs.py is for the case where unigram and Trigram HMM models in which rare words are subcategorized into 4 separate classes.

To get the tags for the case where words are categorized only as rare, do the following:
1. Run the file using the command python count_freqs_baseline.py gene.train
2. This will generate two text files gene_dev_baseline.p1.out and gene_dev.viterbi_baseline which will be having the words and their corresponding tags for the unigram and Trigram HMM models for the case where rare words are categorized
only in one class.
3. The corresponding tags can be evaluated using the command python eval_gene_tagger.py gene.key gene_dev_baseline.p1.out
                                                             python eval_gene_tagger.py gene.key gene_dev.viterbi_baseline

To get the tags for the case where rare words are subcategorized into 4 classes, do the following:
1. Run the file using the command python count_freqs.py gene.train
2. This will generate two text files gene_dev.p1.out and gene_dev.viterbi which will be having the words and their corresponding tags for the unigram and Trigram HMM models for the case where rare words are categorized
into 4 subcategories.
3. The corresponding tags can be evaluated using the command python eval_gene_tagger.py gene.key gene_dev.p1.out
                                                             python eval_gene_tagger.py gene.key gene_dev.viterbi


To get the tags for the case where rare words are subcategorized into 4 classes and laplace smoothing is also done, do the following:
1. Run the file using the command python count_freqs_laplace.py gene.train
2. This will generate a text file gene_dev.viterbi_laplace which will be having the words and their corresponding tags Trigram HMM model for the case where rare words are categorized
into 4 subcategories with laplace smoothing.
3. The corresponding tags can be evaluated using the command python eval_gene_tagger.py gene.key gene_dev.viterbi_laplace
                                                             


