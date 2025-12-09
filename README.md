# Introduction

**Team Leader:** Yelena Shabanova (320991)  
**Team Members:** Alena Seliutina (323591), Luis Fernando Henriquez Patino (314661)

In this project, we explored the Audience Decode task using unsupervised machine learning techniques. Report of our work will follow the numerical order that corresponds with the `main.ipynb` sections. The provided dataset  contains detailed rating histories for each user, summary statistics for both users and movies, and additional movies' attributes. Our goal was to explore and model audience behavior. Moreover, we aimed to identify meaningful groups of users, detect unusual behavioral patterns, and compare how different clustering algorithms perform on high-dimensional, noisy data.

We started by performing an exploratory data analysis (EDA) on both users and movies. This helped us better understand rating distributions, activity patterns, and relationships between features. We first applied clustering to the movie dataset to create pseudo-genres since the project description mentions "analyzing interaction with genres" but this information is not provided in database. Therefore, we made our definition of content types based on features given in movie data. They later helped us interpret user preferences.

Next, we clustered the users using three methods: K-Means, DBSCAN and BIRCH. Each model has different strengths, but after comparing the results, we chose K-Means as our main model, because it produced the detailed and most stable clusters. These user clusters represent different types of audience behavior and allow us to estimate their pseudo-genre preferences.

Finally, we looked at how these preferences changed over time to understand how audience behavior evolves. Overall, our project shows how unsupervised learning can help discover hidden patterns in user interactions and provide insights into audience segmentation.

---

# Methods

After managing the libraries and loading all data by connecting to the `viewer_interactions.db`, we load the key tables into pandas DataFrames. We got:

- `user_statistics`: Pre-computed behavioral data, serving as the main features for each user.  
- `movie_statistics`: Pre-computed statistical data for each movie.  
- `movies`: Essential metadata for movies, including the title and year of release.

Then we started from exploratory data analysis, continuing through preprocessing and clustering, and ending with model selection. Below, we describe each methodological step, along with the tools used and the outputs obtained. All figures referenced in this section are placed in an additional folder named `images/`, which accompanies the README.

---

## 3. Exploratory Data Analysis (EDA)

We must have understood the model behaviour. Therefore, we performed an EDA to analyze the distributions of our data. For this part we mainly used `pandas`, `numpy`, `matplotlib`, and `seaborn`, as well as `sqlite3` to extract data from the original database.

### 3.1 User EDA

In this section, using `user_statistics`, our goal was to understand viewer activity patterns and the distributions of key behavioral features. Then we were able to cluster users based on shared patterns. We examined distributions of:

- total number of ratings per user  
- users’ average rating  
- standard deviation of ratings  
- activity levels  

We showed that on visualizations (Figures U1–U7).

First, we displayed a summary table of the main user-level metrics (Figure U1). This table provides an overview of how users behave on the platform in terms of activity and rating style.

**Figure U1 – User statistics summary**  
[images/user_statistics_summary.csv](images/user_statistics_summary.csv)

**Interpretation (U1):**

- `total_ratings` is highly right-skewed: most users rate only a few titles, while a small minority rate dozens or hundreds.  
- `avg_rating` varies widely across users, indicating different rating styles (harsh vs. generous raters).  
- `std_rating` shows how consistent a user is in their scoring behavior.  
- `unique_movies` closely mirrors `total_ratings`, showing that users rarely re-rate the same film.  
- `activity_days` captures long-term engagement over time.

Next, we plotted a histogram of `total_ratings` (Figure U2) to understand how active users are. This plot shows that user activity is extremely imbalanced, with a large passive group and a small highly active group. Activity becomes a key dimension for clustering.

**Figure U2 – Histogram of total ratings per user**  
![U2 – user total ratings](images/user_total_ratings_hist.png)


- Most users rate very few movies: the majority give between 1 and 5 ratings.  
- Around 75% of users rate fewer than 11 movies, showing a large group of low-activity users.  
- Moderately active users (20–50 ratings) form a smaller but visible segment.  
- There is a long tail of “power users” with 100+ ratings, but they are rare.

Then we looked at the distribution of users’ average rating using a histogram (U3) and a violin plot (U4). Together, these plots show that rating style (how generous or harsh a user is) is another important behavioral dimension we need to capture.

**Figure U3 – Histogram of average user rating**  
![U3 – user avg rating histogram](images/user_avg_rating_hist.png)


- Most users have an average rating between 3.0 and 4.0, showing a general positive bias.  
- There are visible spikes at integer values, as ratings are discrete (1–5).  
- Some users systematically give low or high scores, meaning rating style varies across the population.

**Figure U4 – Violin plot of average user rating**  
![U4 – user avg rating violin](images/user_avg_rating_violin.png)


- The densest region is around 3.5–4.0, confirming most users tend to rate positively.  
- Thin tails at the extremes indicate small groups of consistently harsh or consistently generous users.  
- The distribution is not perfectly symmetric, and its shape suggests substantial variation in rating style.

To study how activity influences consistency, we grouped users into activity quartiles based on `total_ratings` and compared their rating behavior (Figure U5). This shows that more active users tend to be more consistent in their ratings. Activity level, therefore, is not only about how much users watch but also about how stable their rating behavior is (again supporting its use as a key feature for clustering).

**Figure U5 – Rating behaviour by activity level**  
![U5 – user activity vs rating](images/user_activity_vs_rating_violin.png)


- Mean ratings stay pretty similar across activity levels, so activity does not significantly change the average score.  
- Low-activity users show very wide variation in average ratings, including many extreme values.  
- Medium and high-activity users are more concentrated around the 3–4 range.  
- Very high-activity users show the narrowest spread, indicating the most stable rating behavior.

Then, we plotted a scatterplot of `unique_movies` vs `total_ratings` using a sample of users (Figure U6) to check for redundancy between these features. As these features are so tightly correlated, we can treat them as essentially encoding the same information. This is important when deciding which variables to keep in the feature matrix.

**Figure U6 – Unique movies vs total ratings**  
![U6 – unique vs total ratings](images/user_unique_vs_total_ratings_scatter.png)


- Points lie very close to a curved line, indicating a near-linear relationship between `unique_movies` and `total_ratings`.  
- This confirms that users almost never re-rate the same movie.  
- The log-scaled axis reveals both low-activity and high-activity users in the same plot.

Finally, we created a correlation heatmap of the main user features (Figure U7). From it we concluded that we should avoid including redundant features in the clustering (`total_ratings` and `unique_movies`), and combine activity, consistency, and rating style features to best capture different user personas.

**Figure U7 – Correlation heatmap of user features**  
![U7 – user features correlation](images/user_features_correlation_heatmap.png)


- Strong correlations appear between activity-related features (`total_ratings` and `unique_movies`).  
- Other features (such as `avg_rating` and `std_rating`) provide additional, less correlated information about rating style and consistency.

All these steps in User EDA represented our later feature selection and clustering design: we decided to focus on a compact set of user features that represent activity, rating style, and rating consistency, while avoiding redundant variables.

All figures U1–U7 mentioned above are stored in the `images/` folder and referenced from the README.

---

### 3.2 Movie EDA

To understand the behavior of movies in the platform and prepare for later movie clustering, we analyzed the `movie_statistics` and `movies` tables. Similar to the user-side EDA, the goal was to understand distribution patterns, find meaningful content-related features and find relationships that could affect clustering performance.

We again used `sqlite3` to retrieve the data and `pandas`, `numpy`, `matplotlib`, and `seaborn` for the visual analysis.

We did it through average movie ratings, rating count (popularity), rating variability, relationship between popularity and rating, distribution of release years and correlation between movie-level features.

We began with analyzing a table of aggregated movie-level statistics (Figure M1). This table already suggests that movies differ strongly across multiple dimensions, which justifies clustering them into data-driven pseudo-genres.

**Figure M1 – Movie statistics summary**  
[images/movie_statistics_summary.csv](images/movie_statistics_summary.csv)

**Interpretation (M1):**

- `avg_rating` ranges widely, showing films receive both very high and very low evaluations.  
- `rating_count` varies drastically, indicating that some movies are rated frequently while others receive very few ratings.  
- `std_rating` captures controversial films (high variance) vs. consistently liked/disliked movies.  
- `year_of_release` spans many decades, meaning the dataset mixes both old and modern titles.

Then, we looked at `rating_count` using a histogram (Figure M2). The goal here was to understand how often movies are rated and whether popularity can help separate different types of films. Popularity ends up being a key dimension for clustering, because it distinguishes extremely niche titles from mid-popular movies and true blockbusters.

**Figure M2 – Movie popularity distribution**  
![M2 – movie popularity tiers](images/movie_popularity_tiers_bar.png)

It showed that popularity is heavily skewed. Most movies have only 1–2 ratings, which already covers the majority of the catalog. After that, the numbers drop sharply: only a small number of movies reach the 6–50 ratings range, showing that mid-popular films are rare.

Popularity rises again only for a very small group of well-known titles. A few hundred movies fall into the 51–1,000 ratings tiers, around two hundred movies reach the 1,000–10,000 ratings range, and only about 80 blockbusters exceed 10,000 ratings, with the most popular title having 173,598 ratings.

This long-tail structure makes it clear why `rating_count` is essential for clustering. It helps us separate almost unseen films, moderately known films, and widely recognized blockbusters. Including popularity ensures that pseudo-genres reflect not only content characteristics but also how widely each film is viewed.

Next, we plotted a histogram of the average movie ratings (Figure M3) to see how movies are generally rated. This helped us see how movies are rated in general and whether this feature separates different types of films.

**Figure M3 – Average movie rating distribution**  
![M3 – movie avg rating hist](images/movie_avg_rating_hist.png)

The distribution is slightly right-skewed: most movies have an average rating between 2.5 and 4.0, while very low-rated movies (below 1.5) or high-rated (above 4.5) exist but are rare. Peaks at integer values occur because movies with very few ratings often have an average equal to their single rating. The range of values is wide enough to show real differences in movie quality, so `avg_rating` is a useful feature for clustering.

To understand how controversial movies are, we looked at the distribution of `std_rating` (Figure M4). This feature helps understand which are stable favorites and which are divisive films.

**Figure M4 – Movie rating variability distribution**  
![M4 – movie std rating hist](images/movie_std_rating_hist.png)

Most movies have a standard deviation around 0.7–1.3, meaning moderate disagreement. A small group has very high variability (>1.2), which suggests controversial or polarizing films. Some movies have very low variability, meaning viewers rate them almost the same way. This confirms that `std_rating` adds important information about how consistent or inconsistent movie ratings are.

We plotted `avg_rating` vs `rating_count` (Figure M5) to understand the relationship between quality and popularity.

**Figure M5 – Popularity vs average rating**  
![M5 – movie popularity vs avg rating](images/movie_popularity_vs_avg_rating_scatter.png)

We found that very popular movies usually have moderately high ratings, and movies with very low ratings almost never become popular. At the same time, low-popularity movies show a wide range of average ratings – typical for niche films. Because the relationship is not purely linear, both features put unique information to clustering.

Next we plotted `std_rating` vs `rating_count` (Figure M6). It shows how rating variability relates to popularity.

**Figure M6 – Rating variability vs popularity**  
![M6 – movie controversy vs popularity](images/movie_controversy_vs_popularity_scatter.png)

It could be seen that highly popular movies usually have lower variance, meaning people generally agree about them. Also movies with high `std_rating` tend to be less popular, therefore controversial films are often niche. Moreover, some unpopular movies show extremely high disagreement, indicating unstable reception. This confirms that popularity and controversy describe different movie behaviors, and both matter for clustering.

Finally, we computed a correlation matrix (Figure M7) for key movie features. Selected features (quality, popularity, variability, and release year) together represent meaningful axes of movie behavior and should all be kept for clustering.

**Figure M7 – Correlation heatmap of movie features**  
![M7 – movie features correlation](images/movie_features_correlation_heatmap.png)


- `rating_count` has moderate correlation with `std_rating` and weak correlation with `avg_rating`, meaning popularity is not simply tied to quality.  
- `avg_rating` and `std_rating` are only weakly related, indicating both provide distinct information.  
- `year_of_release` has almost no correlation with the other features, meaning movie age adds its own independent information.

We concluded that movies differ a lot in quality, popularity, variability, and age, so grouping them into pseudo-genres makes sense. Overall, movies differ clearly in how well they are rated, how many people rate them, how consistent those ratings are, and how old they are. Because of this, clustering them into pseudo-genres is a reasonable next step before analyzing user preferences.

All figures M1–M7 are stored in the `images/` folder and referenced in the README.

---

## 4. Preprocessing and Feature Engineering

After completing the EDA, we prepared both user and movie data for clustering. Since our methods (KMeans, DBSCAN, BIRCH) are distance-based, they require a consistent set of numeric features, no missing values and features on comparable scales. To achieve this, we built two standardized feature matrices:

- `X_users_kmeans` – standardized user-level behavioral features  
- `X_movies_kmeans` – standardized movie-level features  

These matrices are the direct input to the clustering models used in Sections 5 and 6.

### 4.1 User Feature Matrix

From the User EDA, we saw that viewers differ mainly in:

- Activity level  
- Average rating style (harsh vs. generous)  
- Consistency of ratings  
- Duration of engagement  

Accordingly, we built the user feature matrix from the following variables taken from `user_statistics`:

- `total_ratings` – total number of ratings per user  
- `unique_movies` – number of distinct movies rated  
- `avg_rating` – user’s mean rating  
- `std_rating` – standard deviation of ratings (rating consistency)  
- `activity_days` – span of user activity in days  

We first extracted these columns into a dedicated DataFrame and inspected missing values. Most missingness came from structural reasons (`std_rating` undefined when a user rated only one movie or partial activity information for very sparse users). We then imputed missing values using a median imputer and standardized all features using `StandardScaler`, so each column has mean 0 and unit variance.

This produced the final user feature matrix `X_users_kmeans`, which contains one standardized behavioral vector per user and no missing values. Although `total_ratings` and `unique_movies` are highly correlated (as shown in the User EDA), we kept both during preprocessing to maintain consistency with earlier steps and let later clustering design decide on redundancy.

### 4.2 Movie Feature Matrix

On the movie side, EDA showed that films differ along four main dimensions:

- Popularity  
- Perceived quality  
- Controversiality / rating variability  
- Age (release year)  

We therefore built the movie feature matrix from:

- `total_ratings` – number of ratings per movie (popularity)  
- `avg_rating` – mean movie rating (quality)  
- `std_rating` – variability of ratings (controversial vs. consistently rated)  
- `year_of_release` – movie age  

These columns were extracted from `movie_statistics` and `movies` into a separate DataFrame. Missing values followed clear patterns: movies with no interactions lacked `total_ratings` and `avg_rating`. Movies with a single rating lacked `std_rating`. Some movies had missing `year_of_release` due to incomplete metadata.

As with users, we imputed all missing values with the median for each feature. Then, we standardized all columns with `StandardScaler`, ensuring that variables like `total_ratings` (potentially very large) do not dominate smaller-scale variables such as `avg_rating`.

The result is the movie feature matrix `X_movies_kmeans`, a clean, fully numeric and standardized representation of every movie used in the clustering methods of Section 5.

### 4.3 Summary of Preprocessing Decisions

We came to several conclusions across both users and movies:

- Missing values were systematic, not accidental: no ratings then no `avg_rating` / `total_ratings`; single rating then missing `std_rating`; incomplete metadata then missing `year_of_release`.  
- Because of this structure, we applied median imputation, which is stable for skewed distributions and fits well with KMeans.  
- No rows were removed: all users and movies are retained after preprocessing.  
- After imputation and scaling, both feature matrices contain zero missing values.  

We got final matrices ready for clustering: `X_users_kmeans` and `X_movies_kmeans`. In both matrices, missing values in features like `std_rating` were replaced using the median and all features were scaled to mean 0 and unit variance. These matrices were then prepared for use in Section 5 (Movie Clustering) and Section 6 (User Clustering).

---

## 5. Movie Clustering (Pseudo-Genre Discovery)

Before clustering users, we first clustered movies into pseudo-genres, since user interpretation later depends on movie grouping. For this section, we used scikit-learn’s KMeans and PCA for visualization.

To identify behavior-based categories of movies, we applied unsupervised clustering to the standardized movie feature matrix constructed earlier. This analysis is entirely driven by user interaction patterns, not by data such as plot or genre. For this section we used scikit-learn’s KMeans and PCA for visualization.

The movie feature matrix included many behavioral attributes such as:

- Popularity: `total_ratings`  
- Perceived quality: `avg_rating`  
- Rating inconsistency: `std_rating`  
- Age of the film: `year_of_release`  

After standardization, these variables form the input for clustering.

### 5.1 Determining Number of Clusters

We evaluated several values of k using: the Elbow Method (Figure MC1), Silhouette Scores (Figure MC2). Both evaluation criteria suggested k = 6 as a reasonable choice, balancing compactness and interpretability.

**Figure MC1 – Movie KMeans elbow (inertia)**  
![MC1 – movie elbow](images/movie_kmeans_elbow_inertia.png)

**Figure MC2 – Movie KMeans silhouette scores**  
![MC2 – movie silhouette scores](images/movie_kmeans_silhouette_scores.png)

We evaluated the number of clusters k using two standard methods:

1. **Elbow Method (Inertia):**  
   Inertia decreases rapidly for low k and begins flattening around k = 6.

2. **Silhouette Score:**  
   Measures how distinct and well-separated clusters are. Silhouette scores were highest for k = 6 within the range tested (k = 2–10).

In the end both diagnostics supported our decision to choose k = 6 as number of clusters, because it balances:

- cluster separation,  
- stability,  
- and interpretability of results.

### 5.2 Training and PCA Visualization

We selected k = 6 as the optimal number of movie clusters based on earlier evaluation. A K-Means model was then initialized with this value and trained on the standardized movie features.

During training, the algorithm learned six behavioral centroids and assigned each movie to the nearest one. The resulting cluster labels were added back to the movie DataFrame, allowing us to interpret each movie pseudo-genre and inspect whether the groups aligned with the expected rating and popularity patterns.

To visually inspect cluster separation, we reduced dimensionality to two principal components using PCA. The scatterplot (Figure MC3) displayed well-formed cluster boundaries, confirming that the chosen features were informative.

**Figure MC3 – PCA scatterplot of movie clusters**  
![MC3 – movie clusters PCA](images/movie_clusters_pca_scatter.png)

To visualize high dimensional clustering results, we applied a two component PCA. The resulting 2D scatterplot (using a sample of 8k movies for better clarity):

- Revealed well-separated regions corresponding to clusters  
- Confirmed that the chosen features carry meaningful variance for grouping movies  
- Provided visual support for k = 6 being a structurally coherent segmentation  

PCA was used strictly for visualization; clustering was performed on the standardized full feature space.

### 5.3 Interpreting the Pseudo-Genres

We summarized each movie cluster using descriptive statistics (Figure MC4) and manually interpreted their characteristics, making the final pseudo-genres:

**Figure MC4 – Movie cluster summary table**  
[images/movie_cluster_summary.csv](images/movie_cluster_summary.csv)

1. **Old Classics:** Older films with moderate engagement and moderate scores  
   - Avg Popularity: ~90 ratings  
   - Avg Rating: 3.05  
   - Rating Variability: ~1.09  
   - Mean Year: 1958  

2. **Well Liked Niche Films:** Films with small audience but very positive reception  
   - Avg Popularity: ~18 ratings  
   - Avg Rating: 4.09  
   - Rating Variability: ~0.46  
   - Mean Year: 1997  

3. **Controversial Films:** Films that divide viewers due to high variability  
   - Avg Popularity: ~202 ratings  
   - Avg Rating: 3.05  
   - Rating Variability: 1.77  
   - Mean Year: 1995  

4. **Blockbuster Hits:** Extremely high popularity with stable ratings; dominant mass-market successes  
   - Avg Popularity: ~99,310 ratings  
   - Avg Rating: 3.76  
   - Rating Variability: ~0.98  
   - Mean Year: 1996  

5. **Popular Films:** Films with strong engagement and consistent ratings  
   - Avg Popularity: ~25,233 ratings  
   - Avg Rating: 3.55  
   - Rating Variability: ~1.02  
   - Mean Year: 1992  

6. **Hated Invisible Film:** Films that were unnoticed and with almost no audience  
   - Avg Popularity: ~2.5 ratings  
   - Avg Rating: 1.48  
   - Rating Variability: ~0.55  
   - Mean Year: 1996  

---

# 3. Experimental Design

## 6. User Clustering

Once pseudo-genres were established, we clustered users based on their engagement level and rating style to uncover behavior-based viewer types.

The standardized user feature matrix included:

- Activity level: `total_ratings`, `unique_movies`, `activity_days`  
- Rating style: `avg_rating`, `std_rating`  

We evaluated multiple clustering algorithms:

- K-Means on the full user dataset  
- DBSCAN on the full standardized matrix  
- BIRCH with behaviour-weighted features  

### 6.1 K-Means: Choosing the Number of User Clusters

We applied the same methods as for movies: Elbow plot for inertia (Figure UC1); Silhouette scores.

**Figure UC1 – User KMeans elbow (inertia)**  
![UC1 – user elbow](images/user_kmeans_elbow_inertia.png)

To choose the number of clusters for K-Means, we:

- Took a random sample from the standardized feature matrix: `X_users_kmeans`.  
- For each k ∈ {2,…,10} we: fit K-Means on the sample and computed the silhouette score and inertia (Elbow method).  
- Additionally, we used `KElbowVisualizer` on the full matrix to confirm elbow region (Figure UC2).

**Figure UC2 – Distortion Score Elbow for K-means Clustering**
![UC2 - distortion score](images/user_kmeans_distortion_score.png)

- The silhouette score was highest at k = 2, but that solution is too rough. Among all k > 2, k = 6 achieved the best silhouette score and lay in the elbow region of the inertia curve, while also yielding richer, more interpretable segments. Therefore we selected k = 6 as the number of user clusters for K-Means.

### KMeans: Final User Segmentation

With k = 6 fixed, we trained K-Means on the full standardized user dataset:

- **Input features:** `total_ratings`, `unique_movies`, `avg_rating`, `std_rating`, `activity_days` (all standardized)  
- **Model:** `KMeans(n_clusters=6, random_state=42, n_init=10)`  
- **Output:** each user received a cluster label  

We then:

- Computed cluster-level summaries (mean, median, count of all five features) to characterize each segment.  
- Projected users into 2D with PCA and plotted a scatterplot colored by cluster label to visually inspect separation.

**User PCA plot (KMeans clusters)**  
![User clusters PCA 2D](images/user_clusters_2D_PCA.png)

This resulted in six distinct behavioral user types, later given descriptive names. These profiles are used later to interpret engagement patterns and preferences.

### 6.2 DBSCAN Analysis

To explore density-based structure, we applied DBSCAN directly on the full standardized user matrix.

DBSCAN produced:

- One dominant cluster containing ≈ 95% of users  
- Several smaller clusters (on the order of 8k–10k users)  
- A number of tiny micro-clusters (< 300 users)  
- About 0.5% of users labeled as noise (Figure DB1)  

**Figure DB1 – DBSCAN clusters (PCA projection)**  
![DB1 – user clusters DBSCAN PCA](images/user_clusters_dbscan_PCA.png)

We then:

- Sampled up to 30,000 users  
- Applied PCA(2) for visualization  
- Plotted the DBSCAN labels in 2D  
- Computed a silhouette score on ~50,000 non-noise users (≈ 0.12)  

(DB2 is not shown as a figure.)

This confirmed that, although DBSCAN reveals some density-based structure, the data is dominated by a large homogeneous core, and the resulting segmentation is not as clean or interpretable as K-Means.

### 6.3 BIRCH Analysis

Considering all models, we decided to experiment with BIRCH due to how it handles big data.

**Choosing k for BIRCH:**

- Used a sample of up to 50,000 users (`X_birch_sample`) from the standardized matrix.  
- Evaluated k = 2…10 using silhouette scores.  
- While k = 2 had the highest silhouette, it collapsed almost all users into one broad cluster and isolated only heavy users, which is not very informative.  
- k = 4 offered a better trade-off between score and interpretability (Figure BCH1).

**Figure BCH1 – BIRCH silhouette scores**  
![BCH1 – user BIRCH silhouette](images/user_birch_silhouette_scores.png)

**Behaviour-weighted BIRCH:**

To focus on behavior rather than just volume, we reweighted features:

- `total_ratings`, `unique_movies` × 0.5 (down-weight volume)  
- `avg_rating`, `std_rating`, `activity_days` × 2.0 (up-weight rating style and temporal behaviour)  

On this behaviour-weighted matrix (`X_birch`), we trained:

- `Birch(n_clusters=4)` on the full dataset  
- Summarized each cluster and visualized them using PCA(2) (Figure BCH2)

**Figure BCH2 – BIRCH clusters (PCA projection)**  
![BCH2 – user clusters BIRCH PCA](images/user_clusters_birch_pca_scatter.png)

This produced four interpretable audience patterns, distinguishing long-time users, critical short-term users, long-term enthusiasts and positive casual raters in a more behavior-focused way than the unweighted version.

### 6.4 Evaluation of Clustering Quality

We compare these models based on silhouette score as evaluation metric. The silhouette score measures how well defined the clusters are. Score ranges from -1 to 1:

- Clusters that are dense and very far apart (the ideal) → +1  
- Clusters that overlap → 0  
- Points that are probably in the wrong cluster → -1  

**K-Means (k = 6)**  
Silhouette Score: ~0.33

K-Means performs strongly and consistently because:

- It creates balanced clusters, instead of one dominating cluster.  
- The segments are easy to interpret: engagement level, rating style, and activity duration emerge naturally.  
- It works directly on the standardized feature matrix with no need for weighting or tuning.  
- It handles the large, high-dimensional dataset efficiently.  
- The resulting clusters align well with human-interpretable behavioral categories (e.g., power users, one-time visitors).

In practice, this makes K-Means the most stable and intuitive method for segmenting users.

**DBSCAN**  
Silhouette Score: ~0.2 (computed only on non-noise users)

DBSCAN struggles with this dataset because user data is very high-dimensional and DBSCAN works best in low dimensional spaces. Moreover, the dataset is highly uneven, with many low-activity users and few heavy users—DBSCAN treats this as one giant dense region.  

As a result, DBSCAN produces:

- One massive cluster containing ~95% of all users  
- Several tiny clusters  
- A large amount of noise  

We understood that user behavior does not create clear density-separated groups. Instead, most users fall into the same broad behavioural region, and the few unusual users only form small, unstable clusters.
DBSCAN helps confirm the overall shape and density of the dataset, but it does not produce meaningful or interpretable user segments.
It gives less stable and useful clustering results, than K-means and Birch.

**BIRCH (k = 4, with behaviour-weighted features)**  
Silhouette Score: ~0.48 (highest)

BIRCH performs well after applying behavior-weighted features, producing four clear audience groups; broad distinct patterns in activity, viewing diversity, and rating stability; and a compact, well-separated cluster structure  

However, BIRCH creates broader and more general clusters, making it less detailed for behavioral segmentation compared to K-Means. It is valuable as a complementary view, but not as the main model for downstream analysis.

**Final Decision: K-Means is Selected**

Although BIRCH achieves a higher silhouette score, we chose K-Means as the final user clustering model for several reasons:

- It creates six well-balanced and interpretable audience segments  
- Clusters reflect meaningful behavioral traits
- The algorithm is computationally efficient and easy to scale  
- It provides exact, stable clusters needed for analyzing genre preferences and long-term engagement

Therefore, K-Means with k = 6 is selected as the main model for all final analyses.
DBSCAN and BIRCH are included only as comparison methods to show how alternative clustering approaches behave on the same data.

An overall silhouette comparison is summarized in comparison table of models' scores:

[images/user_clustering_model_scores.csv](images/user_clustering_model_scores.csv)


### Interpretion of User Clusters for a chosen K-means model

After we finished K-Means, we had to reference user groups in later sections, so we assigned descriptive behavioral labels. 
The names and parameters that were given are:

- **Cluster 0 — Consistent Enjoyers**  
  - ~29 total ratings, ~29 unique movies  
  - Positive average rating (~3.6)  
  - Activity span around 447 days (1+ year)  
  - Steady users who explore many movies and show consistent engagement (≈ 42k users)

- **Cluster 1 — Fan Visitors**  
  - ~2–3 total ratings, ~2 unique movies  
  - Very high average rating (~4.5)  
  - Short activity (mean ~64 days, median 0)  
  - Brief visitors who rate a couple of favourite movies generously and leave (≈ 101k users)

- **Cluster 2 — One-Movie Critics**  
  - ~1–2 total ratings, usually 1 unique movie  
  - Average rating ~2.3–2.5  
  - Very short activity (~30 days)  
  - Users who appear once, rate a movie or two, and do not return (≈ 49k users)

- **Cluster 3 — Loyal Occasional Enjoyers**  
  - ~14 ratings, ~12 unique movies  
  - Balanced average rating (~3.56)  
  - Very long activity (~1060 days ≈ 3 years)  
  - Loyal long-term users who come back over years but do not rate heavily (≈ 46k users)

- **Cluster 4 — Heavy Enthusiasts**  
  - ~78+ ratings, 65+ unique movies  
  - Average rating (~3.38)  
  - Long activity (~900 days)  
  - High-engagement users who explore many movies and remain active for years (≈ 6k users)

- **Cluster 5 — Short-Term Users**  
  - ~6–7 ratings, ~5 unique movies  
  - Neutral average rating (~3.5)  
  - Activity ~180 days  
  - Viewers who interact lightly and stay for only several months (≈ 170k users)

These clusters represent distinct audience types and are later used to analyze:

- How different user types consume the movie pseudo-genres  
- How engagement and rating behaviour evolve over time  
- How viewing patterns differ across segments in terms of intensity, generosity, and stability  



## 7. User–Movie Preferences

This section unites two major components of our project:

- **Audience Segments:** discovered through user clustering, where 487,780 users were matched to one of the six user clusters derived
  
- **Content Groups:** groups derived from movie clustering, where 16,013 movies were matched to one of the six behavioral genres 

To understand how different audiences engage with different types of content and how this behavior changes over time, we analyze the `viewer_ratings` interaction table, which contains over four million ratings. Because loading this table all at once would exceed available memory, the analysis is designed to extract only necessary columns and load it in chunks, allowing us to clearly observe the evaluation and efficiency of the data.

### 7.1 Merge of Users and Movie Clusters

Before analyzing preferences, we combined three tables:

- viewer ratings (`customer_id`, `movie_id`, `rating, date`)  
- user cluster labels  
- movie cluster labels  

We create simple lookup dictionaries, which contain a map between each user/movie and their respective cluster. Then we start iterating through the ratings table in chunks of 400k rows. For each chunk we attach the audience segment and movie genre to each rating; extract the year the rating was made and store the partial summaries and yearly activities for later.  

This allowed us to understand which audience segments watch which pseudo-genres, and how this changes over time.

### 7.2 The Build of the Matrices

After we merged users, movies, and interactions, the next step was to quantify how each user segment engages with each movie genre. To do this, we constructed three complementary matrices that summarize viewing behavior and preferences.

**Methods Used:**

For each pair *(`user_clusters`, `movie_clusters`)*, we compute:

- `n_ratings` - how many ratings this audience segment gave to that content type  
- `avg_rating` - how they rate that content type on average  

Based on these values together we build 3 complementary matrices:


- **Preference Matrix (`pref_mean`)**:
  [images/average_rating_summary.csv](images/average_rating_summary.csv)

The table reports the average rating each user segment assigns to each genre. This allows us to compare how positively different audiences evaluate different types of content. 


- **Count Matrix (`pref_count`)**   
 
  [images/number_ratings_summary.csv](images/number_ratings_summary.csv)

The table reports how many ratings each user cluster assigned to each genre, showing how strongly different audience segments engage with various content types.


- **Engagement Share Matrix (`engagement_share`)** 

  [images/share_genre_in_user_segment.csv](images/share_genre_in_user_segment.csv)

This table reports the proportion of each genre within the viewing activity of each user cluster. It shows what types of movies each user segment actually spends their time watching, regardless of rating behavior.


To summarize above mentioned matrices, we built 2 heatmaps: 
1. Heatmap of average ratings based on preference matrix
2. Heatmap of rating frequencies based on engagement share matrix 

They will be interpreted in the next section.


### 7.3 Temporal Analysis Design

To analyze how behavior changes over time, date-time in `viewer_ratings` are converted to years.

For each year, we computed:

- number of ratings given by each user cluster  
- number of ratings received by each genre 

We then converted these into yearly shares. This allowed us to compute how activity of users changes and the popularity of genres over years. Findings are shown and interpreted in the next section by graphs.

---

# Results

This section represents the main findings of the project after computing every single section that has been mentioned before. The results summarize how different audience segments engage with different types of movies, how positively they rate them and how their viewing patterns change in scale and structure.

By combining the movie clusters and the user clusters with the full interaction history, we obtained a complete picture of the platform audience environment, revealing which users prefer which genres, which groups are most active and how tastes differ across segments.

First, we provide an interpretation of our results through heatmaps and graphs that we discussed in previous section.

## Heatmaps Interpretation

### Share of Genres Watched Heatmap
This heatmap shows the proportion of each genre within each user segment’s total viewing activity.

![EXP4 – share of genres watched](images/share_genres_watched_by_users.png)

- Mainstream genres dominate viewing for nearly all user groups. Bright blue blocks under Blockbusters and Popular Films (0.52 for Fan Visitors, 0.43 for Consistent Enjoyers) show that these categories absorb most platform activity.

- Heavy Enthusiasts appear more exploratory. Their row contains noticeably lighter cells for Old Classics (0.08) and Well-Liked Niche Films (0.04), indicating above-average engagement with less mainstream content.

- One-Movie Critics show a unique viewing anomaly. The lightest cell in the Hated Invisible Films column (0.06) appears in their row, unlike all other clusters. This shows disproportionate consumption of fringe or obscure titles.

- Short-Term Users and Consistent Enjoyers share similar viewing patterns. Their rows have nearly identical blue intensity across Blockbusters and Popular Films, showing strong mainstream focus.

### Average Ratings Given Heatmap
This heatmap displays the average rating each user cluster assigns to each movie genre.

 ![EXP3 – average rating heatmap](images/average_rating_user_segment_genre.png)

- Fan Visitors stand out with the brightest (yellow) row.
 Their ratings cluster around 4.44–4.48, showing extremely positive evaluations across all genres.

- One-Movie Critics form the darkest (purple) row.
 Ratings such as 1.55 (Hated Invisible Films) and 2.19–2.66 across other categories show consistently negative evaluation patterns. 

- Consistent Enjoyers and Short-Term Users display stable mid-green shades.
 Their rows remain consistently around 3.6–3.8, showing balanced, predictable, moderate positivity.

- Heavy Enthusiasts, sit in the middle band.
 Their ratings (3.10–3.67) are neither too low nor too high, matching their broad viewing behavior.

- Loyal Occasional Enjoyers show moderately warm colors in mainstream genres like Blockbusters (3.72) and Well-Liked Niche Films (3.77), showing generally positive attitudes toward common content.


## Graphs Interpretation
### User Cluster Activity Over Time

This plot shows how the share of total user activity (ratings) changed for each user group from 1995 to 2006.

  ![EXP5 – user cluster activity over time](images/user_total_share_over_time.png)


- Early years (1995–1999): One-Movie Critics dominate. 
At the beginning of the platform, most activity came from users who only rated one or two movies. Their share is the highest in the late 1990s.

- Around 2000–2001: A spike in Loyal Occasional Enjoyers.
Suddenly, long-term but moderate users become the main group. Their activity share jumps dramatically for a short period, meaning many casual users joined.

- 2002 onward: Shift toward broader, more balanced usage. 
Loyal Occasional Enjoyers decline again as the user base grows and becomes more mixed.

- 2002–2005: Strong rise of Short-Term Users.
A new wave of users appears (people who use the platform briefly, rate a few movies, and then leave). Their share becomes one of the biggest during these years.

- Consistent Enjoyers also increase after 2002.
Mid-activity, stable users grow steadily and peak around 2004. They represent a more engaged and persistent audience.

- Fan Visitors remain small but steady. 
They contribute consistently but never dominate, which matches their behavior (rating only a couple of favourites).

- Heavy Enthusiasts stay rare. T
heir share is always low, but slowly increases over time—meaning “super-active” users exist, but they’re a small niche.

  
### Movie cluster (genres) popularity over time 

This graph shows how much each user group contributed to all ratings over time. It helps us to see which audience types were most active in different years.

![EXP6 – movie cluster popularity over time](images/movie_cluster_popularity_over_time.png)

- Early years (1995–1999): Blockbusters dominate.
At the start of the platform, the majority of ratings go to Blockbusters and other highly popular movies. Their share is the highest in the late 1990s, showing that early users mostly engaged with mainstream, well-known titles.

- Around 2000–2001: A spike in Well-Liked Niche Films.
The share of Well-Liked Niche Films rises sharply, becoming briefly the most rated category. This suggests that many users were discovering mid-popular, high-quality films during this period—possibly due to catalogue expansion or changing user interests.

- 2002 onward: Popular Films gain momentum.
After 2001, the share of Popular Films (mid-mainstream titles) grows steadily, indicating a shift toward more balanced and diverse movie consumption as the platform matures.

- 2002–2005: Strong rise of Hated or Invisible Films.
Surprisingly, the share of Hated Invisible Films increases noticeably during 2003–2005. This likely reflects the arrival of casual users who browse widely, try random low-rated films, and contribute scattered ratings.

- Classic Films also increase after 2002. 
Old Classics begin to attract more attention, peaking around mid-2000s. This shows growing interest in older, culturally important movies among the expanding user base.

- Blockbusters remain stable but no longer dominant.
Although mainstream films stay popular throughout the timeline, they gradually lose their early dominance as the platform becomes more diverse in terms of movie preferences.

# Project Discoveries

Uniting all of our analysis and results, we now can **decode the Audience**.
Here we interpret every discovery that has been found to make the results clear and readable. 
This section contains all about user behavior, viewing preferences and temporal trends to make  a clear audience interpretation and understand how each segments interacts with different genres.

## What Each User Segment Watches

**Consistent Enjoyers (Cluster 1)**

- Behavior: Steady, moderate–high engagement. 
- Viewing: Mostly Popular Films (43%) and Blockbusters (30%). 
- Taste: Positive ratings (~3.7–3.8).

**Insight**: A stable, satisfied core audience focused on mainstream titles.

**Fan Visitors (Cluster 2)**

Behavior: Very low activity, extremely generous ratings.
Viewing: Blockbusters (52%), Popular Films (28%).
Taste: Highest ratings of all clusters (4.4+).
Insight: One-time visitors drawn to trending movies; high enthusiasm, no retention.

**One-Movie Critics (Cluster 3)**

- Behavior: Minimal activity, harsh scoring.
- Viewing: Blockbusters (47%), Popular Films (20%), Controversial Films (18%), and uniquely Invisible Films (6%).
- Taste: Lowest ratings across all genres (1.5–2.6)

**Insight**: Niche explorers who rarely enjoy what they watch.

**Loyal Occasional Enjoyers (Cluster 4)**

- Behavior: Very long-term users with light but steady activity. 
- Viewing: Popular Films (38%), Blockbusters (36%). 
- Taste: Moderate positivity (≈3.5–3.7).

**Insight**: Loyal but infrequent viewers who prefer mainstream content.

**Heavy Enthusiasts (Cluster 5)**
- Behavior: Highest engagement; wide exploration. 
- Viewing: More diverse: Controversial Films (34%), Popular Films (37%), Classics (8%). 
- Taste: Balanced, realistic ratings (~3.4–3.5).

**Insight**: Deep catalog explorers enriching the platform’s depth.


**Short-Term Users (Cluster 6)**
- Behavior: Short-lived engagement; neutral ratings. 
- Viewing: Blockbusters (48%), Popular Films (34%). 
- Taste: Positive for mainstream, lower for niche.

**Insight**: Trial-phase users who mostly consume promoted films.


## How Each User Segment Rates Content
- Fan Visitors inflate all ratings (4.4+).
- Consistent Enjoyers, Loyal Occasionals, and Heavy Enthusiasts anchor the platform with realistic positive ratings (3.4–3.8).
- Short-Term Users, resembles enjoyers but drops sharply on niche categories.
- One-Movie Critics rates all clusters poorly

**Insight**: Rating style clearly separates stable users from temporary or dissatisfied ones.


## Temporal Behavior of User Segments

- 1999–2001: Loyal Occasional Enjoyers temporarily dominate activity.
- Post-2001: Consistent Enjoyers and Heavy Enthusiasts become the primary long-term groups.
- Fan Visitors and Short-Term Users decline over time.
- One-Movie Critics remain small.

**Insight**: The user base shifts toward more stable, high-engagement profiles.

## Temporal Behavior of Movie Clusters

- Popular Films and Blockbusters rise and remain dominant after 1999–2000.
- Niche and Invisible Films decline.
- Old Classics remain small but present.
- Controversial Films stay mid-tier.

**Insight**: Consumption shifts increasingly toward mainstream genres.

## Combined Segment-Genre Matching

- Consistent enjoyers = Popular FIlms and Blockbusters
- Fan visitors = Blockbusters and popular films 
- Loyal Occasionals = popular films and blockbusters 
- Heavy enthusiast = controversial and invisible films 
- One movie critics = Controversial and invisible films 
- Short term users = blockbusters and popular films

## Overall

Mainstream genres dominate nearly all viewer groups.
Enthusiasts and Critics mainly prefer Niche and controversial categories.
User behavior clearly separates long-term, valuable users from short-term ones.
Consistent Enjoyers, Loyal Occasionals, and Heavy Enthusiasts - are the platform’s and most influential audience groups.
Fan Visitors give high ratings but engage very little, while One-Movie Critics and Short-Term Users contribute minimal or irregular activity.
Over time, stable user groups become more important and viewing becomes increasingly centered on popular genres.

# Conclusion

To conclude, even though our work focuses on how specific types of users interact with different types of movies, 
grouping movies into pseudo-genres and users into behavioural segments, provides a clear analysis of the engagement, habits and patterns among any type of data.

Still, some questions remain open. While our analysis reveals clear behavioural patterns, our project does not expand on how these user segments would react to new or upcoming releases. 

This is essential area to strengthen future path toward building effective and accurate movie recommendation systems. 
We could explore how users discover new data, as well as how new coming users would be placed into a proper audience segment.
We believe, that natural next step would be to integrate temporal patterns and simple content features, allowing recommendations to adapt to evolving tastes and better predict how each behavioural group might engage with newly released movies.
This whole project and its high interpretation allow us to build a strong foundation for possible future work in recommendations.

