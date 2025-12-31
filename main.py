# Task - Task 3 – Predicting the product category based on the title

# Exercise
# Task context
# Imagine you are part of the development team of an e-commerce company that introduces thousands of new products into the system every day. The real challenge: every product needs to be categorized correctly and quickly, but manual classification consumes valuable time and increases the risk of error.

# To make the process more efficient, the team decided to develop an intelligent system that would automatically suggest the appropriate category based on the product title entered by the user.

# You are responsible for developing this model – your solution will be an essential part of the system that, every day, makes the work of hundreds of colleagues easier and improves the user experience on the platform.

 

# Task objective
# The goal of this assignment is to develop a machine learning model that automatically suggests the right category for each new product, based on its title. This will help make the process of introducing products to the online platform faster, simpler, and more accurate – without manual classification and with a lower risk of error.

# Your model allows each new article to be immediately assigned to the correct category, speeding up the entire team’s work, making search easier, and improving the user experience on the site. At the same time, you gain experience in creating and applying an ML solution to a concrete business challenge as part of a broader digital ecosystem.

 

# Why is this task important?
# Automatic product classification is not just a technical innovation – it is key to the operational efficiency of any modern online store. Manually classifying thousands of new products is time-consuming, increases the risk of error, and slows down the entire process.

# By developing this model, you show how machine learning skills can solve a concrete and real problem – speed up team work, reduce costs, and provide a better user experience.

# At the same time, you learn how to manage a complete ML project: from understanding business requirements, to preparing data, to implementing a solution that the team can use immediately.

# This task gives you the opportunity to demonstrate how your knowledge can have a visible and positive impact on daily work and user satisfaction – exactly as expected from a data specialist in a modern business environment.

 

# What's at your disposal? – Your toolbox for a complete ML project
# The data set

# You have a real and rich data set at your disposal (products.csv) with over 30,000 products from different categories.

# For each product you receive:

# Product ID – unique identifier;
# Product Title – product title (for example, "Samsung Galaxy A52 128GB");
# Merchant ID – seller;
# Category Label – the target category (e.g., “Mobile Phones”, “Laptops”);
# Product Code – cod intern;
# Number of Views – number of views;
# Merchant Rating – seller rating;
# Listing Date – listing date.
# Testing products

# After you develop and test the model, you can also check it "manually" - enter one of these titles and see how the model reacts:



# Your pregnancy
# Your task is to develop and distribute a complete project for automatically classifying products into categories, using a real dataset.

# The project must contain:

# The trained model, saved in .pkl format ;
# a Python script for training and saving the model;
# a Python script for interactive testing: the user enters the product title, and the model predicts the category;
# Jupyter notebooks with complete analysis, feature engineering, model training and evaluation;
# a public GitHub repository with the entire project, including:
# all relevant scripts and notebooks;
# Clear README with instructions for using and testing the model;
# code documentation and logical project organization.
# Note: Each component is expected to be clearly documented and prepared for further use or development within the team.

 

# Hint 
# This is a task where you are in the role of a real ML developer:

# There is no “right way” – the important thing is to explore, experiment, and document your work so that your team can easily pick up where you left off.

# Tips for a successful job:

# There's no need for everything to be perfect the first time – focus on a clear workflow and clear communication of results.
# Feel free to use the official documentation (pandas, scikit-learn, Git, pickle...) and resources on StackOverflow or GitHub when you encounter difficulties, as well as the lessons in the course.
# Experiment with different approaches: try different models, different features , text vectorization methods…
# Every key decision, attempt, “ fail ” or pivot – document it in your notebook or README file ! This is gold for your team.
# If you don't know something – ask the instructor or write down the question as part of the documentation. Real development teams progress exactly like this!
 

# Roadmap for solving the task
# Approach this task like a true member of the data team – every step brings you closer to a solution that the entire team can use immediately!

# Create a GitHub repository
# Publish the project, as you would in a real team (name, description, initial README ).
# 2. Clone the local repository

# If you want to work locally, ensure synchronization between your computer and the remote repository.
# 3. Create one or more notebooks (Colab/Jupyter)

# This is where you experiment, analyze, and process data.
# 4. Load and explore data

# Understand the structure of the data set and identify potential problems or inconsistencies.
# 5. Prepare and clean the data

# Resolve missing values, standardize data, and prepare it for modeling.
# 6. Explore the possibilities of feature engineering

# Try different features – each addition should have a purpose and potential value for the model.
# 7. Compare the performance of multiple ML algorithms

# Test different models and find the best balance between accuracy, robustness, and interpretability.
# 8. Train and save the final model

# Choose the best solution, train the final model, and save it for later use.
# 9. Create regular checkpoints/commits

# After each major step, commit – it's important that your work is transparent and easy to track in the project history.
# 10. Create the final scripts:

# train_model.py – the logic for training and saving the model (based on your analysis and testing).
# predict_category.py – logic for loading the model and interactive testing (entering the product title and getting the category).
# 11. Organize the project and add README.md

# The repository structure must be logical, and the README clear and useful for any team member.
# Finally, make sure that the project can be used by any team member without further explanation – this is a sign of professional ownership!

# Resources & help

# If you're struggling or want to explore further, use these guides as a starting point – and write down your best insights and mini-fails for the team!

# Feature engineering
# Don't limit yourself to just " Product Title " – experiment with additional features ( feature engineering ) that can improve the model:

# the number of words or characters in the title;
# the presence of numbers or special characters;
# if the title contains the brand name or all capital letters (USB, LED…);
# the length of the longest word...
# These "small" features can be essential for more accurate classification!

# Recommendation: Document observations – what makes sense, what didn’t work, so that the entire team understands your process.

# Model evaluation
# Evaluate the model using the following metrics:

# accuracy ;​
# classification report ( precision , recall , F1-score by categories);
# the confusion matrix (ideally visualized – helps the team see where the model is wrong).
 

# Handing over the task
# At the end of the task, your public GitHub repository should contain:

# the dataset ( products.csv ) that you used to train the model;
# at least one .ipynb notebook with complete analysis and solution development (or multiple notebooks with clearly separated phases);
# Python scripts: train_model.py (for training) and predict_category.py (for interactive model testing);
# a clear and well-organized README.md file with instructions for running and testing the project.
# How do you teach the task?

# Check that the repository is public and complete.
# Review the README – can any user run the project by following the instructions?
# Send the link to your GitHub repository to the course instructor through the form provided on the learning platform.
# Quick check before handover:

# Is each step of the analysis clearly presented and commented on with your logic?
# Can someone continue working or use your model immediately without further explanation?
# Does the code (notebooks and scripts) work without errors when run "from scratch"?
# Is the project structure clear, and is the documentation useful and easy to understand?
# Before handing over, ask yourself: Can I, in a month, without consulting additional materials, easily continue developing based on this project?

