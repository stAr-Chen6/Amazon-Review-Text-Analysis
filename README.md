# Amazon-Review-Text-Analysis





Folder "Code":  
1. Data Preprocessing.ipynb:  
        -Get data about “sweatshirt” from the category "Clothing_Shoes_and_Jewelry";
        -extract the data that the review part has more than 100 characters.  


2. Criteria Extraction and Evaluation.ipynb:
        -All code to extract the six criteria:  
                i) LDA analysis method;                                                                              
		ii)TF-IDF with K-Means Clustering method. 
        -All code to evaluate the six chosen criteria and show the importance level. 


3. Method1.ipynb:  
        -All code to apply method 1 and evaluation. 


4. Method2.py:  
        -All code to apply method 2:   
                i) TF-IDF vectorizer;                                                                             
		ii)predicting partial scores;                                                                                      		
		iii)computing R^2 and MAE between predicted overall score and ground truth. 
                                                                                                 
5. Method3.ipynb:  
        -All code to apply method 3.  






Folder "Data":  
1.Method 1:  
        -negative_word.txt:Globally defined negative words for Word Embedding	
        -positive_words.txt:Globally defined positive words for Word Embedding	


2.Method 2
        -method2_significance_analysis.xlsx: Student T-test between method 2 and baseline (random labeling)


3.Method 3
        -method3_significance_analysis.xlsx: Student T-test between method 3 and baseline (random labeling)


4.Method Comparison
        -method_performance.xlsx:Data of four different metrics that can evaluate the performance of three different methods.
        -method_comparison.xlsx:Student T-test between method 2 and method 3.


5.shirt_meta.csv:
        -All meta-data such as title about the products whose name contains “shirt”.


6.shirt_review.csv:
        -All data about the products whose name contains “shirt”.


7.sweatshirt.csv:
        -All reviews about sweatshirt from shirt_review.csv


8.sweatshirt_100words.csv:
        -All reviews with more than 100 characters from sweatshirt.csv


9.sweatshirt_sample1200.csv:
        -1200 reviews extracted from sweatshirt_100words.csv for manually labeling


10.1200.csv:
        -Manually labeled dataset for training models and evaluation


11.sweatshirt_total.csv:
        -Data used for method 2, containing both manually labelled data and reviews without labels
