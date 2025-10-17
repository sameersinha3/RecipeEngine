import pandas as pd
import numpy as np
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi


class FoodComDataCollector:    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.api = KaggleApi()
        self.api.authenticate()
        
        os.makedirs(data_dir, exist_ok=True)
    
    def download_dataset(self) -> bool:
        try:
            print("Downloading Food.com dataset from Kaggle...")
            kaggle.api.dataset_download_files(
                'irkaal/foodcom-recipes-and-reviews',
                path=self.data_dir,
                unzip=True
            )
            print("Dataset downloaded successfully!")
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        recipes_path = os.path.join(self.data_dir, "recipes.csv")
        reviews_path = os.path.join(self.data_dir, "reviews.csv")
        
        if not os.path.exists(recipes_path) or not os.path.exists(reviews_path):
            raise FileNotFoundError(
                "Dataset files not found. Please run download_dataset() first."
            )
        
        print("Loading raw data...")
        recipes_df = pd.read_csv(recipes_path)
        reviews_df = pd.read_csv(reviews_path)
        
        print(f"Loaded {len(recipes_df)} recipes and {len(reviews_df)} reviews")
        return recipes_df, reviews_df
    
    def parse_ingredients(self, ingredients_str: str) -> List[str]:
        if pd.isna(ingredients_str):
            return []
        
        try:
            # Parse R language format: c("ingredient1", "ingredient2", ...)
            if ingredients_str.startswith('c(') and ingredients_str.endswith(')'):
                # Remove c( and ) wrapper
                content = ingredients_str[2:-1]
                # Split by commas, but be careful about commas inside quotes
                ingredients = []
                current_ingredient = ""
                in_quotes = False
                
                for char in content:
                    if char == '"' and (not current_ingredient or current_ingredient[-1] != '\\'):
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        # End of current ingredient
                        ingredient = current_ingredient.strip().strip('"')
                        if ingredient:
                            ingredients.append(ingredient)
                        current_ingredient = ""
                    else:
                        current_ingredient += char
                
                # Add the last ingredient
                if current_ingredient.strip():
                    ingredient = current_ingredient.strip().strip('"')
                    if ingredient:
                        ingredients.append(ingredient)
                
                # Clean and normalize ingredients
                cleaned_ingredients = []
                for ingredient in ingredients:
                    cleaned = re.sub(r'\s+', ' ', ingredient.strip())
                    if cleaned:
                        cleaned_ingredients.append(cleaned)
                return cleaned_ingredients
        except:
            pass
        
        return []
    
    def parse_steps(self, steps_str: str) -> List[str]:
        if pd.isna(steps_str):
            return []
        
        try:
            # Parse R language format: c("step1", "step2", ...)
            if steps_str.startswith('c(') and steps_str.endswith(')'):
                # Remove c( and ) wrapper
                content = steps_str[2:-1]
                # Split by commas, but be careful about commas inside quotes
                steps = []
                current_step = ""
                in_quotes = False
                
                for char in content:
                    if char == '"' and (not current_step or current_step[-1] != '\\'):
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        # End of current step
                        step = current_step.strip().strip('"')
                        if step:
                            steps.append(step)
                        current_step = ""
                    else:
                        current_step += char
                
                # Add the last step
                if current_step.strip():
                    step = current_step.strip().strip('"')
                    if step:
                        steps.append(step)
                
                # Clean and normalize steps
                cleaned_steps = []
                for step in steps:
                    cleaned = re.sub(r'\s+', ' ', step.strip())
                    if cleaned:
                        cleaned_steps.append(cleaned)
                return cleaned_steps
        except:
            pass
        
        return []
    
    def parse_tags(self, tags_str: str) -> List[str]:
        if pd.isna(tags_str):
            return []
        
        try:
            # Parse R language format: c("tag1", "tag2", ...)
            if tags_str.startswith('c(') and tags_str.endswith(')'):
                # Remove c( and ) wrapper
                content = tags_str[2:-1]
                # Split by commas, but be careful about commas inside quotes
                tags = []
                current_tag = ""
                in_quotes = False
                
                for char in content:
                    if char == '"' and (not current_tag or current_tag[-1] != '\\'):
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        # End of current tag
                        tag = current_tag.strip().strip('"')
                        if tag:
                            tags.append(tag)
                        current_tag = ""
                    else:
                        current_tag += char
                
                # Add the last tag
                if current_tag.strip():
                    tag = current_tag.strip().strip('"')
                    if tag:
                        tags.append(tag)
                
                # Clean and normalize tags
                cleaned_tags = []
                for tag in tags:
                    cleaned = re.sub(r'\s+', ' ', tag.strip().lower())
                    if cleaned:
                        cleaned_tags.append(cleaned)
                return cleaned_tags
        except:
            pass
        
        return []
    
    def parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse ISO 8601 duration string (PT45M, PT2H30M, etc.) to minutes."""
        if pd.isna(duration_str) or duration_str == '':
            return None
        
        try:
            duration_str = str(duration_str).strip()
            if not duration_str.startswith('PT'):
                return None
            
            # Remove PT prefix
            duration_str = duration_str[2:]
            total_minutes = 0
            
            # Parse hours (H)
            if 'H' in duration_str:
                hours_part = duration_str.split('H')[0]
                total_minutes += int(hours_part) * 60
                duration_str = duration_str.split('H')[1]
            
            # Parse minutes (M)
            if 'M' in duration_str:
                minutes_part = duration_str.split('M')[0]
                if minutes_part:  # Make sure it's not empty
                    total_minutes += int(minutes_part)
            
            return total_minutes if total_minutes > 0 else None
        except:
            return None
    
    def process_recipes(self, recipes_df: pd.DataFrame, 
                       reviews_df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        print("Processing recipe data...")
        
        if limit is not None:
            recipes_df = recipes_df.head(limit)
            print(f"Processing only first {limit} recipes for testing...")
        
        processed_recipes = []
        
        for _, recipe in tqdm(recipes_df.iterrows(), total=len(recipes_df)):
            ingredients = self.parse_ingredients(recipe['RecipeIngredientParts'])
            steps = self.parse_steps(recipe['RecipeInstructions'])
            tags = self.parse_tags(recipe['Keywords'])
            
            # Parse cooking times from ISO 8601 duration format
            prep_time = self.parse_duration(recipe['PrepTime'])
            cook_time = self.parse_duration(recipe['CookTime'])
            total_time = self.parse_duration(recipe['TotalTime'])
            
            processed_recipe = {
                'recipe_id': recipe['RecipeId'],
                'title': recipe['Name'].strip() if pd.notna(recipe['Name']) else '',
                'ingredients': ingredients,
                'instructions': steps,
                'tags': tags,
                'prep_time': prep_time,
                'cook_time': cook_time,
                'total_time': total_time,
                'n_steps': len(steps),
                'n_ingredients': len(ingredients),
                'rating': recipe['AggregatedRating'] if pd.notna(recipe['AggregatedRating']) else None,
                'review_count': recipe['ReviewCount'] if pd.notna(recipe['ReviewCount']) else None,
                'author_id': recipe['AuthorId'],
                'author_name': recipe['AuthorName'],
                'category': recipe['RecipeCategory'],
                'description': recipe['Description'],
                'date_published': recipe['DatePublished']
            }
            
            processed_recipes.append(processed_recipe)
        
        return pd.DataFrame(processed_recipes)
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = "processed_recipes.csv") -> str:
        filepath = os.path.join(self.data_dir, filename)
        
        # Convert lists to strings for CSV compatibility
        df_csv = df.copy()
        df_csv['ingredients'] = df_csv['ingredients'].apply(
            lambda x: '|'.join(x) if x else ''
        )
        df_csv['instructions'] = df_csv['instructions'].apply(
            lambda x: '|'.join(x) if x else ''
        )
        df_csv['tags'] = df_csv['tags'].apply(
            lambda x: '|'.join(x) if x else ''
        )
        
        df_csv.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        return filepath
    
    def save_to_json(self, df: pd.DataFrame, filename: str = "processed_recipes.json") -> str:
        filepath = os.path.join(self.data_dir, filename)
        
        # Convert DataFrame to list of dictionaries
        recipes_list = df.to_dict('records')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(recipes_list, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {filepath}")
        return filepath
    
    def run_full_pipeline(self, output_format: str = 'both', limit: Optional[int] = None) -> Dict[str, str]:
        print("Starting Food.com data collection pipeline...")
        
        # Download dataset
        if not self.download_dataset():
            raise Exception("Failed to download dataset")
        
        recipes_df, reviews_df = self.load_raw_data()
        
        processed_df = self.process_recipes(recipes_df, reviews_df, limit=limit)
        
        output_files = {}
        
        if output_format in ['csv', 'both']:
            csv_path = self.save_to_csv(processed_df)
            output_files['csv'] = csv_path
        
        if output_format in ['json', 'both']:
            json_path = self.save_to_json(processed_df)
            output_files['json'] = json_path
        
        print(f"\nPipeline completed! Processed {len(processed_df)} recipes.")
        print(f"Output files: {list(output_files.values())}")
        
        return output_files
