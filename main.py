import argparse
import sys
from data_collector import FoodComDataCollector


def main():
    
    parser = argparse.ArgumentParser(
        description="Collect and preprocess recipe data from Food.com dataset"
    )
    
    parser.add_argument(
        '--output-format',
        choices=['csv', 'json', 'both'],
        default='both',
        help='Output format for processed data (default: both)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Directory to store downloaded and processed data (default: data)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (use existing files)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of recipes to process (useful for testing)'
    )
    
    args = parser.parse_args()
    
    try:
        collector = FoodComDataCollector(data_dir=args.data_dir)
        
        if args.skip_download:
            print("Skipping download, using existing data...")
            recipes_df, reviews_df = collector.load_raw_data()
            processed_df = collector.process_recipes(recipes_df, reviews_df, limit=args.limit)
            
            output_files = {}
            if args.output_format in ['csv', 'both']:
                csv_path = collector.save_to_csv(processed_df)
                output_files['csv'] = csv_path
            
            if args.output_format in ['json', 'both']:
                json_path = collector.save_to_json(processed_df)
                output_files['json'] = json_path
        else:
            output_files = collector.run_full_pipeline(output_format=args.output_format, limit=args.limit)
        
        print("\n" + "="*60)
        print("RECIPE DATA COLLECTION COMPLETE!")
        print("="*60)
        print(f"Output files created:")
        for format_type, filepath in output_files.items():
            print(f"  {format_type.upper()}: {filepath}")
        
        if args.skip_download:
            recipes_df, reviews_df = collector.load_raw_data()
            processed_df = collector.process_recipes(recipes_df, reviews_df, limit=args.limit)
        else:
            import pandas as pd
            if 'json' in output_files:
                import json
                with open(output_files['json'], 'r') as f:
                    processed_df = pd.DataFrame(json.load(f))
            else:
                processed_df = pd.read_csv(output_files['csv'])
        
        print(f"\nDataset Statistics:")
        print(f"  Total recipes: {len(processed_df):,}")
        print(f"  Recipes with ingredients: {processed_df['n_ingredients'].gt(0).sum():,}")
        print(f"  Recipes with instructions: {processed_df['n_steps'].gt(0).sum():,}")
        print(f"  Recipes with tags: {processed_df['tags'].apply(len).gt(0).sum():,}")
        print(f"  Recipes with cooking time: {processed_df['total_time'].notna().sum():,}")
        print(f"  Average ingredients per recipe: {processed_df['n_ingredients'].mean():.1f}")
        print(f"  Average steps per recipe: {processed_df['n_steps'].mean():.1f}")
        print(f"  Average cooking time: {processed_df['total_time'].mean():.1f} minutes")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
