import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import joblib

class CricketAnalyticsEngine:
    """
    Main analytics engine that processes cricket data and generates insights
    """
    
    def __init__(self, data_directory="CRICKET_DATA"):
        self.data_dir = data_directory
        self.datasets = {}
        self.processed_data = None
        self.results = {
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'version': '1.0',
                'datasets_loaded': 0
            },
            'overview': {},
            'performance_prediction': {},
            'clustering': {},
            'format_analysis': {},
            'business_insights': {}
        }
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
    def load_datasets(self):
        """Load all cricket datasets using your exact file paths"""
        print("🏏 Loading Cricket Datasets...")
        
        try:
            # Load datasets using your exact paths
            self.datasets = {}
            
            # Batting data
            self.datasets['batting_odi'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Batting\ODI data.csv')
            self.datasets['batting_t20'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Batting\t20.csv')
            self.datasets['batting_test'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Batting\test.csv')
            
            # Bowling data
            self.datasets['bowling_odi'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Bowling\Bowling_ODI.csv')
            self.datasets['bowling_t20'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Bowling\Bowling_t20.csv')
            self.datasets['bowling_test'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Bowling\Bowling_test.csv')
            
            # Fielding data
            self.datasets['fielding_odi'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Fielding\Fielding_ODI.csv')
            self.datasets['fielding_t20'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Fielding\Fielding_t20.csv')
            self.datasets['fielding_test'] = pd.read_csv(r'C:\Users\nigar\Downloads\cricket_data\Fielding\Fielding_test.csv')
            
            loaded_count = 0
            for key, df in self.datasets.items():
                if df is not None and not df.empty:
                    loaded_count += 1
                    print(f"✅ Loaded {key}: {df.shape}")
                    # Display column info for first dataset to understand structure
                    if loaded_count == 1:
                        print(f"   📋 Columns: {list(df.columns)}")
                        print(f"   📊 Sample data:\n{df.head(2)}")
                        
            self.results['metadata']['datasets_loaded'] = loaded_count
            print(f"\n📊 Successfully loaded {loaded_count}/9 datasets")
            
            return loaded_count > 0
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            print("💡 Make sure the file paths are correct and files exist")
            return False
    
    def process_and_engineer_features(self):
        """Advanced data processing and feature engineering"""
        print("\n🛠️ Processing and Engineering Features...")
        
        all_data = []
        
        for dataset_name, df in self.datasets.items():
            if df is None or df.empty:
                continue
                
            # Extract format and category from dataset name
            parts = dataset_name.split('_')
            category = parts[0]  # batting, bowling, fielding
            format_type = parts[1]  # odi, t20, test
            
            # Create a copy for processing
            processed_df = df.copy()
            
            # Add metadata
            processed_df['category'] = category
            processed_df['format'] = format_type.upper()
            processed_df['dataset_source'] = dataset_name
            
            # Standardize player name column (assuming first column is player name)
            if len(processed_df.columns) > 0:
                processed_df = processed_df.rename(columns={processed_df.columns[0]: 'player_name'})
            
            # Feature engineering based on category
            if category == 'batting':
                processed_df = self._engineer_batting_features(processed_df)
            elif category == 'bowling':
                processed_df = self._engineer_bowling_features(processed_df)
            elif category == 'fielding':
                processed_df = self._engineer_fielding_features(processed_df)
            
            all_data.append(processed_df)
        
        if all_data:
            # Combine all processed data
            self.processed_data = pd.concat(all_data, ignore_index=True, sort=False)
            
            # Additional cross-format features
            self.processed_data = self._engineer_cross_format_features(self.processed_data)
            
            print(f"✅ Processed data shape: {self.processed_data.shape}")
            
            # Generate overview statistics
            self._generate_overview_stats()
            
            return True
        
        return False
    
    def _engineer_batting_features(self, df):
        """Engineer batting-specific features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Basic performance metrics
            df['total_performance'] = df[numeric_cols].sum(axis=1, skipna=True)
            df['avg_performance'] = df[numeric_cols].mean(axis=1, skipna=True)
            df['performance_consistency'] = df[numeric_cols].std(axis=1, skipna=True)
            
            # Batting specific calculations (adapt based on actual columns)
            if 'Runs' in df.columns and 'Innings' in df.columns:
                df['batting_average'] = df['Runs'] / df['Innings'].replace(0, 1)
                df['runs_per_match'] = df['Runs'] / df.get('Matches', df['Innings']).replace(0, 1)
            
            if 'Runs' in df.columns and 'Balls' in df.columns:
                df['strike_rate'] = (df['Runs'] / df['Balls'].replace(0, 1)) * 100
            
            # Performance categories
            if 'Runs' in df.columns:
                df['performance_tier'] = pd.cut(
                    df['Runs'], 
                    bins=[0, 500, 2000, 5000, 10000, float('inf')],
                    labels=['Emerging', 'Developing', 'Established', 'Star', 'Legend']
                )
        
        return df
    
    def _engineer_bowling_features(self, df):
        """Engineer bowling-specific features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            df['total_performance'] = df[numeric_cols].sum(axis=1, skipna=True)
            df['avg_performance'] = df[numeric_cols].mean(axis=1, skipna=True)
            df['performance_consistency'] = df[numeric_cols].std(axis=1, skipna=True)
            
            # Bowling specific calculations
            if 'Wickets' in df.columns and 'Matches' in df.columns:
                df['wickets_per_match'] = df['Wickets'] / df['Matches'].replace(0, 1)
            
            if 'Runs_Conceded' in df.columns and 'Overs' in df.columns:
                df['economy_rate'] = df['Runs_Conceded'] / df['Overs'].replace(0, 1)
            
            if 'Runs_Conceded' in df.columns and 'Wickets' in df.columns:
                df['bowling_average'] = df['Runs_Conceded'] / df['Wickets'].replace(0, 1)
        
        return df
    
    def _engineer_fielding_features(self, df):
        """Engineer fielding-specific features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            df['total_performance'] = df[numeric_cols].sum(axis=1, skipna=True)
            df['avg_performance'] = df[numeric_cols].mean(axis=1, skipna=True)
            df['performance_consistency'] = df[numeric_cols].std(axis=1, skipna=True)
            
            # Fielding specific calculations
            if 'Catches' in df.columns and 'Matches' in df.columns:
                df['catches_per_match'] = df['Catches'] / df['Matches'].replace(0, 1)
        
        return df
    
    def _engineer_cross_format_features(self, df):
        """Engineer features across formats"""
        if 'player_name' in df.columns:
            # Player format versatility
            format_versatility = df.groupby('player_name')['format'].nunique().reset_index()
            format_versatility.columns = ['player_name', 'format_versatility']
            
            df = df.merge(format_versatility, on='player_name', how='left')
            
            # Player category versatility
            category_versatility = df.groupby('player_name')['category'].nunique().reset_index()
            category_versatility.columns = ['player_name', 'category_versatility']
            
            df = df.merge(category_versatility, on='player_name', how='left')
        
        return df
    
    def _generate_overview_stats(self):
        """Generate overview statistics"""
        if self.processed_data is None:
            return
        
        overview = {
            'total_records': len(self.processed_data),
            'unique_players': self.processed_data['player_name'].nunique() if 'player_name' in self.processed_data.columns else 0,
            'formats_analyzed': self.processed_data['format'].nunique() if 'format' in self.processed_data.columns else 0,
            'categories_analyzed': self.processed_data['category'].nunique() if 'category' in self.processed_data.columns else 0,
            'data_completeness': ((self.processed_data.notna().sum().sum() / 
                                 (self.processed_data.shape[0] * self.processed_data.shape[1])) * 100).round(2),
            'format_distribution': self.processed_data['format'].value_counts().to_dict() if 'format' in self.processed_data.columns else {},
            'category_distribution': self.processed_data['category'].value_counts().to_dict() if 'category' in self.processed_data.columns else {}
        }
        
        self.results['overview'] = overview
    
    def run_performance_prediction(self):
        """Run performance prediction analysis"""
        print("\n🤖 Running Performance Prediction Analysis...")
        
        if self.processed_data is None:
            return False
        
        try:
            # Prepare data for modeling
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove metadata columns from features
            feature_cols = [col for col in numeric_cols if not col.startswith('format_') and col not in ['total_performance']]
            
            if len(feature_cols) < 2:
                print("⚠️  Insufficient features for modeling")
                return False
            
            # Use total_performance as target if available, otherwise use first numeric column
            target_col = 'total_performance' if 'total_performance' in numeric_cols else numeric_cols[0]
            feature_cols = [col for col in feature_cols if col != target_col]
            
            # Prepare modeling dataset
            model_data = self.processed_data[feature_cols + [target_col, 'format', 'category']].dropna()
            
            if len(model_data) < 20:
                print("⚠️  Insufficient data for reliable modeling")
                return False
            
            X = model_data[feature_cols]
            y = model_data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            rf_model.fit(X_train, y_train)
            
            # Predictions and metrics
            y_pred = rf_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Save model
            joblib.dump(rf_model, 'results/performance_prediction_model.pkl')
            
            # Store results
            self.results['performance_prediction'] = {
                'model_metrics': {
                    'mse': float(mse),
                    'r2_score': float(r2),
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                },
                'feature_importance': {k: float(v) for k, v in sorted_features},
                'top_features': [item[0] for item in sorted_features[:10]],
                'predictions_sample': {
                    'actual': y_test.head(10).tolist(),
                    'predicted': y_pred[:10].tolist()
                },
                'target_variable': target_col,
                'features_used': feature_cols
            }
            
            print(f"✅ Performance Prediction Model - R²: {r2:.3f}, MSE: {mse:.2f}")
            return True
            
        except Exception as e:
            print(f"❌ Error in performance prediction: {e}")
            return False
    
    def run_clustering_analysis(self):
        """Run player clustering analysis"""
        print("\n🎯 Running Clustering Analysis...")
        
        if self.processed_data is None:
            return False
        
        try:
            # Prepare clustering data
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if not col.startswith('format_')]
            
            if len(feature_cols) < 2:
                print("⚠️  Insufficient features for clustering")
                return False
            
            cluster_data = self.processed_data[feature_cols + ['player_name', 'format', 'category']].dropna()
            
            if len(cluster_data) < 10:
                print("⚠️  Insufficient data for clustering")
                return False
            
            # Standardize features
            scaler = StandardScaler()
            X = cluster_data[feature_cols]
            X_scaled = scaler.fit_transform(X)
            
            # Perform K-means clustering
            n_clusters = min(4, len(cluster_data) // 3)  # Ensure reasonable cluster size
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels
            cluster_data['cluster'] = clusters
            
            # Define cluster names (you can customize these)
            cluster_names = {
                0: 'Power Performers',
                1: 'Consistent Players', 
                2: 'Aggressive Style',
                3: 'Balanced Approach'
            }
            
            # Limit to available clusters
            available_names = {k: v for k, v in cluster_names.items() if k < n_clusters}
            cluster_data['player_type'] = cluster_data['cluster'].map(available_names)
            
            # Analyze clusters
            cluster_stats = {}
            for cluster_id in range(n_clusters):
                cluster_subset = cluster_data[cluster_data['cluster'] == cluster_id]
                cluster_stats[available_names[cluster_id]] = {
                    'count': len(cluster_subset),
                    'avg_performance': cluster_subset[feature_cols].mean().to_dict(),
                    'format_distribution': cluster_subset['format'].value_counts().to_dict() if 'format' in cluster_subset.columns else {},
                    'category_distribution': cluster_subset['category'].value_counts().to_dict() if 'category' in cluster_subset.columns else {}
                }
            
            # Save clustering model
            joblib.dump(kmeans, 'results/clustering_model.pkl')
            joblib.dump(scaler, 'results/clustering_scaler.pkl')
            
            # Store results
            self.results['clustering'] = {
                'n_clusters': n_clusters,
                'cluster_names': available_names,
                'cluster_distribution': cluster_data['player_type'].value_counts().to_dict(),
                'cluster_statistics': cluster_stats,
                'features_used': feature_cols,
                'sample_data': cluster_data[['player_name', 'player_type', 'format']].head(20).to_dict('records')
            }
            
            print(f"✅ Clustering Analysis Complete - {n_clusters} clusters identified")
            return True
            
        except Exception as e:
            print(f"❌ Error in clustering analysis: {e}")
            return False
    
    def run_format_analysis(self):
        """Run format comparison analysis"""
        print("\n📊 Running Format Analysis...")
        
        if self.processed_data is None or 'format' not in self.processed_data.columns:
            return False
        
        try:
            numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Format comparison statistics
            format_stats = {}
            for format_name in self.processed_data['format'].unique():
                format_subset = self.processed_data[self.processed_data['format'] == format_name]
                format_stats[format_name] = {
                    'count': len(format_subset),
                    'avg_metrics': format_subset[numeric_cols].mean().to_dict(),
                    'std_metrics': format_subset[numeric_cols].std().to_dict()}
        except Exception as e:
            print(f"❌ Error in format analysis: {e}")
            return False