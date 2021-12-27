# Amazon-Review-Text-Analysis

## The project aims to analyze customer reviews of sweatshirt products on Amazon.



Folder "Code":  
1. Data Preprocessing.ipynb:  
        -Get data about “sweatshirt” from the category "Clothing_Shoes_and_Jewelry";  
        -extract the data that the review part has more than 100 characters.  


2. Criteria Extraction and Evaluation.ipynb:  
        -All code to extract the six criteria:  
                i) LDA analysis method;                                                                              
		ii)TF-IDF with K-Means Clustering method. 
        -All code to evaluate the six chosen criteria and show the importance level. 


3. Method1.ipynb (achieved by a teammate):  
        -All code to apply method 1 and evaluation. 


4. Method2.py (achieved by a teammate):  
        -All code to apply method 2:   
                i)TF-IDF vectorizer;                                                                             
		ii)predicting partial scores;                                                                                      		
		iii)computing R^2 and MAE between predicted overall score and ground truth.  
                                                                                                 
5. Method3.ipynb:  
        -All code to apply method 3:
		i)TF-IDF vectorizer and word embedding;                                                                             
		ii)predicting partial scores;                                                                                      		
		iii)computing R^2 and MAE between predicted overall score and ground truth. 


10. 1200.csv:  
        -Manually labeled dataset for training models and evaluation.  


11. sweatshirt_total.csv:  
        -Data used for method 2, containing both manually labelled data and reviews without labels.     
