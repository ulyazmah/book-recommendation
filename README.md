# Book Recommendation System

Try this recommendation system: https://ulya-book-recommender.streamlit.app/

## Introduction
This is my end-to-end data science project involving exploratory data analysis, machine learning modeling and web app deployment. In this project, I tried to develop book recommendation system based on data from the Goodreads application. This dataset was originally scraped from the Goodreads API in September 2017 by Zygmunt Zając and updated by Olivier Simard-Hanley. 
You can download the data from this [Github repository](https://github.com/malcolmosh/goodbooks-10k-extended).

## Objective
The main objective of this project is to develop book recommendation systems by several methods and explore the strengths and weaknesses of each method. 
 
## Goals
### A. Exploratory Data Analysis 
In this part, I explore the data to get understanding about the dataset. There are several deep dive questions should be answered:
1. How is the rating for all books distributed? 
2. How is the average rating per user distributed?
3. How many ratings does a book usually get? 
4. How many ratings does a user usually give? 
5. Does the ratings count affect average rating?
6. Which book has the highest rating and which book has the most ratings? 
7. How is the relationship between the number of ratings and the average rating? 
8. Who is the author with most books? 
9. Who is the most popular author? 
10. Who is the author that has good ratings book? 
11. How is the relationship between the number of pages and the year the book was published?
12. What genre dominates the dataset?
	
### B. Modelling 
In this part, I explored several methods, namely:
1. Simple Recommender: This model offers generalized recommendations to every user based on popularity and average rating of the book. 
2. Content Based Filtering: This model suggests books to users based on the characteristics or attributes of the books. Using this data, the machine will measure the similarity of the books  and then suggests books that are most similar to a particular book that a user liked. 
3. Collaborative Filtering: This system suggests book to users based on past reading history and collaboration of other users' preferences. This recommender systems provides personalized recommendation.

## Conclusion
### 1. Simple Recommender
Like the name suggests, this model offers the simple recommendation.  This model does not provide user-spesific recommendations but suitable for new user (have no cold-start problem)
### 2. Content based filtering
By applying a content based model, instead of having to rate 30 books to start the recommendation engine, users can just pick one book they liked for Goodreads to provide good recommendations for new users, making the process easier. This method is suitable for people who are looking for books that are similar to their favorite books, but this system cannot capture tastes and provide recommendations across genres 
### 3. Collaborative Filtering
Using this method, we can get more personalized recommendation but this model need users  data and not suitable for new users (have cold start problem).
