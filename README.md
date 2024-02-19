# $MealRec^+$

## Contributors
Ming Li, Wuhan University of Technology;

Lin Li, Wuhan University of Technology;

Xiaohui Tao, University of Southern Queensland;

Jimmy	Huang, York University

## Data Description
The suit of $MealRec^+$ datasets we released contains two datasets, $MealRec^+_H$ with a user-meal interaction density of 0.77% and $MealRec^+_L$ with a user-meal interaction density of 0.17%.

Each dataset contains files as follows:
- **relationship information**: 
  - **user_course.txt** : the interaction information between users and courses;
      - format: [user index][\tab][course index]
  - **course_category.txt** : the correspondence information between courses and categories;
      - format: [course index][\tab][category index]
        - category_index = {0:appetizers, 1:main dishes, 2:desserts} 
  - **meal_course.txt** : the affiliation information between meals and courses;
      - format: [meal index][\tab][course index]
  - **user_meal.txt** : the interaction information between users and meals. This file is divided into training set, verification set and test set according to the ratio of 8:1:1. 
      - format: [user index][\tab][meal index]
- **healthiness information**
    - **course_fsa.txt/course_who.txt**: FSA/WHO healthiness scores of courses;
      - format: [healthiness score] *(line index == course index)*
    - **meal_fsa.txt/meal_who.txt**: FSA/WHO healthiness scores of meals;
      - format: [healthiness score] *(line index == meal index)*
    - **user_fsa.txt/user_who.txt**: Mean FSA/WHO healthiness scores of meals that each user has interacted with historically
      - format: [healthiness score] *(line index == user index)*
- **meta information**
  - **course.csv**: meta data of course;
    - fileds: course_id, course_name, review_nums, category, aver_rate, image_url, ingredients, cooking_directions, nutritions, reviews, tags
  - **user_course.csv**: meta data of user-course interaction;
    - fileds: user_id, course_id, rating, dateLastModified
  - **user2index.txt**: a mapping from user id to user index;
    - format: [user id][\tab][user index]
  - **course2index.txt**: a mapping from course id to course index;
    - format: [course id][\tab][course index]

## Data Loader
We provide a data loader (data_loader.py) to load relationship data as matrices and healthiness data as arrays for ease of use. 
