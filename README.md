# MOVIELENS RECOMMENDER SYSTEM
![Movie lens recommender](./image/movies.jpeg)

## Project Overview and Problem Statement
Our goal is to build a system that recommends the top 5 movies to a specific user based on their historical ratings. The core idea is to leverage the collective wisdom of all users in the MovieLens dataset to predict what an individual will enjoy. This problem is tackled by answering four fundaments questions:
 
## Fundamental Questions:
1. How do we find users with the same movie taste? (Collaborative Filtering (User-User))
Method: We calculate a "taste similarity" score between users. The most common way is using Cosine Similarity or Pearson Correlation or average rating.
How it works: Each user is represented as a vector of their movies ratings. We then compare the rating vector of our target user to every other user's vector.
Anology: It's like finding your "movie doppelganger". If you and another user have both rated 20 movies similarly (e.g you both loved the Godfather and hated a particular romantic comedy), the system identifies you as having similar taste. This user's other high-rated movies become strong candidates for your recommendations.
    
2. How do we find movies that are similar to the ones the user already likes? (Content-Based/item-item Collaborative Filtering)
Content-Based: We use metadata about the movies like the genres. If a user highly rates Inception, we recommend other movies tagged as "sci-fiction", "thriller".
Item-Item Collaborative Filtering: It finds similarity based on user behavior. "People who liked movie X also liked movie Y." We compute similarity between movies based on how users have rated them collectively. For example, if users who rated The Shawshank Redemption highly also consistently rated The Godfather highly, these two movies are deemed similar.
    
3. How can we predict a user's rating for an unrated movie?
This is the core prediction engine, by combining the concepts above.
User-User Prediction: To predict your rating for Movie M, the system looks at the users most similar to you (your "neighbors") who have already rated Movie M. Your predicted rating is a weighted average of their ratings for that movie.
Item-Item Prediction: To predict your rating for Movie M, the system looks at the movies most similar to Movie M that you have already rated. Your predicted rating is calculated based on your ratings for these similar movies.
    
4. How do we generate the final Top 5 recommendations?
Once we can predict ratings, then we:
Generate Predictions: For our target user, we run the predictions algorithm for all movies the user has not yet seen.
Rank the list: We sort this list of unseen movies by their predicted rating in descending order.
Select the top N: The top 5 movies from this sorted list are selected as the final recommendations.


# Movies Dataset Overview

## Movies Data
- **Entries**: 9,742 movies
- **Features**:
  - `movieId` (int64): Unique movie identifier
  - `title` (object): Movie title with release year
  - `genres` (object): Pipe-separated list of genres
- **Data Quality**: No missing values in any column

## Ratings Data
- **Entries**: 100,836 ratings
- **Features**:
  - `userId` (int64): Unique user identifier
  - `movieId` (int64): Corresponding movie identifier
  - `rating` (float64): User rating (typically 0.5-5.0)
  - `timestamp` (int64): Unix timestamp of rating
- **Data Quality**: No missing values in any column

## Tags Data
- **Shape**: 3,683 rows × 4 columns
- **Columns**:
  - `userId`: int64 (user identifier)
  - `movied`: int64 (movie identifier)
  - `tag`: object (string tag applied by user)
  - `timestamp`: int64 (Unix timestamp)
- **Data Quality**: None in any column

## Links Data
- **Shape**: 9,742 rows × 3 columns
- **Columns**:
  - `movied`: int64 (movie identifier)
  - `imdbId`: int64 (IMDB identifier)
  - `tmdbId`: float64 (TMDb identifier, stored as float due to missing values)
- **Data Quality**: 
  - 8 missing values in `tmdbId` column
  - None in `movierd` or `imdbId`

## Key Points
- Clean dataset with no missing values other than `tmdbId` which has 8 missing entries out of 9,742
- Structured for movie recommendation analysis
- Contains user-movie interactions with ratings
- Movies dataset includes genre information for content-based filtering

# Models Used
1. Baseline Model
2. SVD Model
3. Hybrid Model

## Baseline Model
### Model Overview
A **simple collaborative filtering** baseline that predicts a rating by combining three core components:

1. **Global Mean (μ)**: The average rating across the entire dataset
2. **User Bias (bᵤ)**: How much a specific user tends to rate above/below the global mean
3. **Movie Bias (bᵢ)**: How much a specific movie is rated above/below the global mean

### Advantages
- **Simple & Interpretable**: Each component has a clear meaning
- **Computationally Efficient**: Only requires storing \(1 + N_{users} + N_{movies}\) parameters
- **Strong Baseline**: Often outperforms naive averages and serves as a foundation for more complex models

### Limitations
- **No Interactions**: Does not capture user–movie preference patterns (e.g., genre preferences)
- **Cold-Start**: For new users/movies, biases default to zero (falls back to global mean)

### Typical Use Case
Used as a starting point in recommendation systems before supplimenting with other models.

## Baseline Model Evaluation & Results

### Model Performance
- **RMSE**: 0.9059 – Lower than all naive baselines (Global Mean, User Mean, Movie Mean)
- **MAE**: 0.6913 – Average prediction error is about 0.69 stars
- **Coverage**: 99.95% – Model can predict ratings for almost all user-movie pairs
- **Improvement**: 13.4% better than simply using the global average rating

**Key Insight**: The baseline model significantly outperforms simple averaging approaches, suggesting it captures meaningful patterns.

### Sample Recommendations (User 429)
**User's Taste Profile**:  
Rated war/action (`Crimson Tide`), space adventure (`Apollo 13`), and animation (`Aladdin`) all as 5 stars.

**Top 5 Recommendations**:  
All predicted as perfect 5.0-star matches:
1. **Paths of Glory** (1957) – War/Drama  
2. **Jules and Jim** (1961) – Drama/Romance  
3. **Yojimbo** (1961) – Action/Drama  
4. **Mr. Death** (1999) – Documentary  
5. **Five Easy Pieces** (1970) – Drama  

**Interpretation**:  
While some recommendations align with war/action themes, others (`Jules and Jim`, documentary) 
seem like exploratory suggestions beyond the users explicit rating history. All perfect 5.0 predictions may indicate
overconfidence or limited rating diversity in training.

## SVD Model (Singular Value Decomposition)
### Overview
A **collaborative filtering** technique that decomposes the sparse user-item rating matrix into lower-dimensional latent factor matrices to uncover hidden patterns and preferences.

### Why It's Used
1. **Captures Latent Features**: Automatically discovers hidden dimensions (e.g., "genre preference", "movie style") without explicit labels
2. **Handles Sparsity**: Works well with sparse rating matrices by learning from patterns across users/items
3. **Enables Personalization**: Predicts ratings for unseen user-item pairs via dot products of latent vectors

### Advantages
- **Strong Predictive Power**: Often outperforms simple bias-based models
- **Efficient Inference**: Once trained, predictions are fast matrix operations
- **Foundation for Extensions**: Basis for many advanced models (SVD++, time-SVD, etc.)

### Challenges
- **Cold-Start Problem**: Cannot handle new users/items without retraining
- **Computational Cost**: Training can be expensive for very large matrices
- **Interpretability**: Latent dimensions are not human-readable

### Common Implementation
Often optimized via **stochastic gradient descent (SGD)** or **alternating least squares (ALS)** to minimize prediction error (e.g., RMSE) on observed ratings.

### Use Case
Well-suited for personalized recommendation systems where sufficient historical interaction data exists, and scalability is important.

# SVD Model Evaluation & Results
### Sample Recommendations (User 429)

## User Profile Overview
- **Movies rated:** 44  
- **Average rating:** 4.07 (high rater)  
- **Top favorites:** Action, Adventure, and Animation (based on `Crimson Tide`, `Apollo 13`, `Aladdin`)

## SVD Recommendations
The **Singular Value Decomposition (SVD)** model predicts ratings based on latent factors from user-item interactions.

1. **Schindler’s List (1993)** – High predicted rating (4.45) aligns with user's preference for drama and historical themes.
2. **Shawshank Redemption (1994)** – Classic drama with strong narrative, similar to user’s top-rated films.
3. **Blade Runner 2049 (2017)** – Sci-Fi pick, possibly influenced by user’s interest in `Apollo 13` (space/tech themes).
4. **Failure to Launch (2006)** – Comedy/Romance outlier; may reflect latent genre diversity in user’s history.
5. **Casablanca (1942)** – Timeless drama/romance; model likely captures classic film appeal.

## Baseline Comparison
Baseline recommendations (e.g., `Paths of Glory`, `Jules and Jim`) appear more **art-house/classic-centric**, possibly based on global popularity or genre trends.

### Key Insight:
SVD recommendations are **more personalized** and reflect the user’s high ratings for mainstream, narrative-driven films, while baseline leans toward critically acclaimed classics with less obvious personal alignment.

# Hybrid Recommendation Strategy

## Hybrid Model Approach
To overcome limitations of single-algorithm systems, we implement a **three-tier strategy**:

1. **SVD Collaborative Filtering** – Primary model for users with sufficient rating history
2. **Content-Based Filtering** – Handles cold-start users (new users or sparse data)
3. **Baseline Model** – Fallback when other models lack confidence

### Benefits:
- Solves **cold-start problem**
- Balances personalization with generalization
- Increases recommendation coverage


## Content-Based Filtering Implementation
We use **TF-IDF on movie genres** to measure similarity between films.

- **TF-IDF Matrix:** `(9742, 24)` – 9,742 movies × 24 genre features
- **Similarity Matrix:** `(9742, 9742)` – Pairwise cosine similarity between all movies

### Example Output:
For *Before and After (1996)* (Genres: `Drama|Mystery`), the model returns films with **identical genre vectors**, resulting in similarity scores of **1.000**.

### Observations:
- **Perfect similarity** occurs when movies share exactly the same genre tags
- This is a **limitation of pure genre-based matching** – diversity may be low
- Future improvements could include **metadata enrichment** (directors, keywords, plot summaries

# Hybrid Recommendation Testing
The hybrid model was tested on two distinct user profiles to demonstrate its adaptive behavior:

## 1. Cold-Start User (User ID: 710)
**Characteristics:** New user with no rating history → classic cold-start problem.

### Hybrid Recommendations:
All recommended movies have the same score (**3.537**), indicating:
- **Content-based or baseline components are dominant**
- Recommendations are **popular, high-rated films** (Forrest Gump, Shawshank Redemption, Pulp Fiction, etc.)
- **Genres are diverse**, providing a broad starting point for user preference discovery

### Strategy Applied:
- Uses **fallback mechanisms** (content-based + baseline)
- Avoids empty recommendations
- Introduces user to widely-acclaimed titles

## 2. Established User (User ID: 298)
**Characteristics:** 
- **842 ratings** → extensive history
- **Average rating: 2.41** → critical rater

### Hybrid Recommendations:
Scores vary (**3.024–2.942**), showing **personalized ranking**:
- Top picks: **Seven Samurai**, **Yojimbo**, **Schindler's List**
- Genres lean toward **Action, Adventure, Drama** with critical acclaim
- Reflects user’s tendency toward **serious, classic, and foreign cinema**

### Strategy Applied:
- **SVD collaborative filtering dominates**
- Recommendations are **tailored** to user’s rich rating profile
- Includes **less mainstream, critically acclaimed films**

# Performance Comparison
The hybrid model was evaluated on a test set using standard regression metrics:

   Model     RMSE      MAE              Description
Baseline 0.905879 0.691330 User + Movie biases only
     SVD 0.918692 0.721764       SVD Implementation
  Hybrid 0.968335 0.762452 SVD + Content + Baseline

### Key Observations:
- **Baseline model performs best** in terms of pure prediction error (lowest RMSE/MAE).
- **Hybrid model has higher error** but offers better **coverage and personalization**.
- **SVD** sits between the two, improving upon baseline for some users while maintaining reasonable error.

## Accuracy vs. Coverage Trade-off
The increase in RMSE/MAE for the hybrid model is expected and acceptable because:

- **Baseline** is simple and generalizes well but lacks personalization.
- **SVD** improves personalization for users with sufficient data.
- **Hybrid** sacrifices some accuracy to **handle cold-start cases** and provide recommendations when pure SVD cannot.

### Practical Implication:
- **Use Baseline** when prediction accuracy is the sole priority.
- **Use Hybrid** when recommendation coverage and user experience (e.g., no empty results) are more important.

## Conclusion
The hybrid model successfully **addresses the cold-start problem** at the cost of a modest increase in prediction error (~5-7% higher RMSE compared to baseline). This is a **reasonable trade-off** for a production system where all users must receive recommendations.
