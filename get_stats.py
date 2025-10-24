import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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

def analyze_data_quality(df):
    """Analyze data quality and completeness"""
    print("\n=== DATA QUALITY ANALYSIS ===")
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    print("Missing data analysis:")
    for col in missing_data.index:
        if missing_data[col] > 0:
            present_count = len(df) - missing_data[col]
            present_percent = 100 - missing_percent[col]
            print(f"  {col}: {present_count:,} present ({present_percent:.2f}%), {missing_data[col]:,} missing ({missing_percent[col]:.2f}%)")
    
    # Data completeness score
    completeness_score = (1 - missing_data.sum() / (len(df) * len(df.columns))) * 100
    print(f"\nOverall data completeness: {completeness_score:.2f}%")
    
    # Rating distribution analysis
    if 'rating' in df.columns:
        rating_stats = df['rating'].describe()
        print(f"\nRating statistics:")
        print(f"  Mean: {rating_stats['mean']:.2f}")
        print(f"  Median: {rating_stats['50%']:.2f}")
        print(f"  Std: {rating_stats['std']:.2f}")
        
        # Rating distribution
        rating_counts = df['rating'].value_counts().sort_index()
        print(f"\nRating distribution:")
        for rating, count in rating_counts.items():
            print(f"  {rating}: {count:,} ({count/len(df)*100:.1f}%)")

def analyze_diversity(df):
    """Analyze dataset diversity and coverage"""
    print("\n=== DIVERSITY ANALYSIS ===")
    
    # Ingredient diversity
    all_ingredients = [ing for ingredients in df['ingredients'] for ing in ingredients]
    unique_ingredients = len(set(all_ingredients))
    total_ingredient_instances = len(all_ingredients)
    
    print(f"Ingredient diversity:")
    print(f"  Unique ingredients: {unique_ingredients:,}")
    print(f"  Total ingredient instances: {total_ingredient_instances:,}")
    
    # Tag diversity
    all_tags = [tag for tags in df['tags'] for tag in tags]
    unique_tags = len(set(all_tags))
    total_tag_instances = len(all_tags)
    
    print(f"\nTag diversity:")
    print(f"  Unique tags: {unique_tags:,}")
    print(f"  Total tag instances: {total_tag_instances:,}")

    # Recipe length diversity
    ingredient_counts = df['n_ingredients'].describe()
    step_counts = df['n_steps'].describe()
    
    print(f"\nRecipe complexity diversity:")
    print(f"  Ingredients per recipe - Mean: {ingredient_counts['mean']:.1f}, Std: {ingredient_counts['std']:.1f}")
    print(f"  Steps per recipe - Mean: {step_counts['mean']:.1f}, Std: {step_counts['std']:.1f}")

def analyze_temporal_distribution(df):
    """Analyze temporal distribution of recipes"""
    print("\n=== TEMPORAL ANALYSIS ===")
    
    if 'date_published' in df.columns:
        df['date_published'] = pd.to_datetime(df['date_published'])
        
        # Year distribution
        df['year'] = df['date_published'].dt.year
        year_dist = df['year'].value_counts().sort_index()
        
        print(f"Recipe distribution by year:")
        print(f"  Date range: {df['date_published'].min().year} - {df['date_published'].max().year}")
        print(f"  Most active year: {year_dist.idxmax()} ({year_dist.max():,} recipes)")
        
        # Decade analysis
        df['decade'] = (df['year'] // 10) * 10
        decade_dist = df['decade'].value_counts().sort_index()
        print(f"\nRecipe distribution by decade:")
        for decade, count in decade_dist.items():
            print(f"  {decade}s: {count:,} ({count/len(df)*100:.1f}%)")

def analyze_bias_and_coverage(df):
    """Analyze potential biases and coverage gaps"""
    print("\n=== BIAS AND COVERAGE ANALYSIS ===")
    
    # Popular ingredients analysis
    all_ingredients = [ing for ingredients in df['ingredients'] for ing in ingredients]
    ingredient_counts = Counter(all_ingredients)
    
    # Calculate concentration (how much of the dataset is dominated by top ingredients)
    top_10_ingredients = ingredient_counts.most_common(10)
    top_10_count = sum([count for _, count in top_10_ingredients])
    concentration_ratio = top_10_count / len(all_ingredients)
    
    print(f"Ingredient concentration:")
    print(f"  Top 10 ingredients represent {concentration_ratio*100:.1f}% of all ingredient instances")
    print(f"  This indicates {'high' if concentration_ratio > 0.3 else 'moderate' if concentration_ratio > 0.2 else 'low'} concentration")


def load_processed_data(path="data/processed_recipes.csv"):
    df = pd.read_csv(path)
    for col in ['ingredients', 'instructions', 'tags']:
        df[col] = df[col].fillna('').apply(
            lambda x: x.split('|') if isinstance(x, str) and x else []
        )
    return df

def preview_embedding_space(df, sample_size=1000):
    sample_size = min(sample_size, len(df))
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
    print("Loading processed data for comprehensive analysis...")
    processed_df = load_processed_data()
    
    print("="*60)
    print("COMPREHENSIVE DATASET ADEQUACY ANALYSIS")
    print("="*60)
    
    # Basic statistics
    dataset_summary(processed_df)
    
    # Data quality analysis
    analyze_data_quality(processed_df)
    
    # Diversity analysis
    analyze_diversity(processed_df)
    
    # Temporal analysis
    analyze_temporal_distribution(processed_df)
    
    # Bias and coverage analysis
    analyze_bias_and_coverage(processed_df)
    
    # Visualization
    print("\nGenerating visualizations...")
    preview_embedding_space(processed_df)

if __name__ == "__main__":
    main()
