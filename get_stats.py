import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def dataset_summary(df):
    print("=== BASIC STATS ===")
    print(f"Total recipes: {len(df)}")
    print(f"Missing titles: {df['title'].isna().sum()}")
    print(f"Avg #ingredients: {df['n_ingredients'].mean():.2f}")
    print(f"Avg #steps: {df['n_steps'].mean():.2f}")
    print(f"Recipes with ratings: {(df['rating'].notna().mean() * 100):.2f}%")
    print(f"Average rating: {df['rating'].mean():.2f}")
    print(f"Recipes with tags: {(df['tags'].apply(len).gt(0).mean() * 100):.2f}%")
    
    print("\n=== TAGS AND INGREDIENTS ===")
    all_tags = [t for tags in df['tags'] for t in tags]
    all_ingr = [i for ing in df['ingredients'] for i in ing]
    
    print(f"Unique tags: {len(set(all_tags))}")
    print(f"Unique ingredients: {len(set(all_ingr))}")
    
    # Show top tags
    top_tags = pd.Series(all_tags).value_counts().head(10)
    print("\nTop 10 tags:")
    print(top_tags)
    
    # Show top ingredients
    top_ing = pd.Series(all_ingr).value_counts().head(10)
    print("\nTop 10 ingredients:")
    print(top_ing)
    
    # Visualizations
    plt.figure(figsize=(6,4))
    sns.histplot(df['n_ingredients'], bins=30)
    plt.title('Distribution of # of Ingredients per Recipe')
    plt.xlabel('Ingredients')
    plt.ylabel('Count')
    plt.show()

    plt.figure(figsize=(6,4))
    sns.histplot(df['rating'].dropna(), bins=20)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Recipes')
    plt.show()

def load_processed_data(path="data/processed_recipes.csv"):
    df = pd.read_csv(path)
    for col in ['ingredients', 'instructions', 'tags']:
        df[col] = df[col].fillna('').apply(
            lambda x: x.split('|') if isinstance(x, str) and x else []
        )
    return df

def preview_embedding_space(df, sample_size=1000):
    sample = df.sample(sample_size, random_state=42)
    texts = sample['title']
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())
    plt.scatter(reduced[:,0], reduced[:,1], alpha=0.5)
    plt.title(f"TF-IDF Embedding Preview ({str(sample_size)} Recipes)")
    plt.show()

def main():
    print("Loading processed data for stats...")
    processed_df = load_processed_data()
    print("Calculating dataset summary...\n")
    dataset_summary(processed_df)
    preview_embedding_space(processed_df)

if __name__ == "__main__":
    main()
