"""
Model Inference Script
Load and use the trained Adult Income Prediction model

Usage:
    python inference.py                    # Run with sample data
    python inference.py --input data.csv   # Run with custom CSV file
"""

import joblib
import pandas as pd
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORTANT: Must define engineer_features function (same as in training)
# The saved model uses FunctionTransformer that references this function
# ============================================================================

def engineer_features(X_in):
    """
    Feature engineering - creates 6 new features from existing ones
    MUST BE IDENTICAL to the function used during training
    """
    X = X_in.copy()
    
    # 1. Age buckets - captures non-linear age effects
    X['age_bucket'] = pd.cut(X['age'], bins=[0, 25, 35, 45, 55, 65, 120],
                              labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'], 
                              right=False)
    
    # 2. Hours buckets - part-time vs full-time vs overtime
    X['hours_bin'] = pd.cut(X['hours-per-week'], bins=[0, 35, 40, 200],
                             labels=['part_time', 'full_time', 'overtime'], 
                             include_lowest=True)
    
    # 3. Capital gain flag - indicates investment income
    X['has_capital_gain'] = (X['capital-gain'] > 0).astype(int)
    
    # 4. Log capital gain - handles skewness
    X['log_capital_gain'] = np.log1p(X['capital-gain'])
    
    # 5. Higher education flag - education level indicator
    X['higher_ed'] = (X['education-num'] >= 13).astype(int)
    
    # 6. Interaction feature - education × hours (captures "educated and works hard")
    X['edu_hours_interaction'] = X['education-num'] * X['hours-per-week']
    
    return X

def load_model(model_path='artifacts/final_pipeline.joblib'):
    """Load the trained model pipeline"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("✓ Model loaded successfully")
    return model

def predict_single(model, age, workclass, education_num, hours_per_week, 
                   capital_gain, occupation, marital_status, sex, race,
                   education='Bachelors', fnlwgt=200000, capital_loss=0,
                   relationship='Husband', native_country='United-States'):
    """Predict for a single person with individual features"""
    
    # Create DataFrame with all required columns
    data = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'education-num': education_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])
    
    # Predict
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0][1]
    
    return pred, proba

def predict_csv(model, csv_path, threshold=0.5):
    """Predict for all rows in a CSV file"""
    df = pd.read_csv(csv_path)
    
    # Ensure all required columns exist
    required_cols = ['age', 'workclass', 'education-num', 'hours-per-week', 
                     'capital-gain', 'occupation', 'marital-status', 'sex', 'race']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Predict
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    predictions_thresholded = (probabilities >= threshold).astype(int)
    
    # Add predictions to DataFrame
    result = df.copy()
    result['prediction'] = predictions
    result['probability'] = probabilities
    result['prediction_thresholded'] = predictions_thresholded
    result['income_class'] = result['prediction_thresholded'].map({0: '<=50K', 1: '>50K'})
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Adult Income Prediction - Model Inference')
    parser.add_argument('--input', type=str, help='Path to CSV file for batch prediction')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold (default: 0.5)')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file for batch predictions')
    parser.add_argument('--model', type=str, default='artifacts/final_pipeline.joblib', help='Path to model file')
    args = parser.parse_args()
    
    print("="*60)
    print("ADULT INCOME PREDICTION - MODEL INFERENCE")
    print("="*60)
    
    # Load model
    model = load_model(args.model)
    
    if args.input:
        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file not found: {args.input}")
            return
        
        # Batch prediction from CSV
        print(f"\nPredicting from: {args.input}")
        try:
            results = predict_csv(model, args.input, args.threshold)
            
            # Summary
            print(f"\n📊 Predictions Summary:")
            print(f"  Total samples: {len(results)}")
            print(f"  Predicted >50K: {(results['prediction'] == 1).sum()}")
            print(f"  Predicted <=50K: {(results['prediction'] == 0).sum()}")
            print(f"  With threshold {args.threshold}: {(results['prediction_thresholded'] == 1).sum()}")
            
            # Distribution
            print(f"\n  Income Distribution:")
            print(results['income_class'].value_counts().to_string())
            
            # Save results
            results.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to: {args.output}")
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return
        
    else:
        # Single prediction with examples
        print("\n📋 Example Predictions:")
        
        # Example 1: High earner
        print("\n  Example 1 (High earner):")
        print("  Age: 50, Education: Doctorate, Hours: 60, Capital Gain: 10000")
        pred, proba = predict_single(
            model,
            age=50,
            workclass='Self-emp-inc',
            education='Doctorate',
            education_num=16,
            hours_per_week=60,
            capital_gain=10000,
            occupation='Exec-managerial',
            marital_status='Married-civ-spouse',
            sex='Male',
            race='White'
        )
        print(f"    → {'>50K' if pred == 1 else '<=50K'} (probability: {proba:.2%})")
        
        # Example 2: Middle earner
        print("\n  Example 2 (Middle earner):")
        print("  Age: 35, Education: Bachelors, Hours: 45, Capital Gain: 5000")
        pred, proba = predict_single(
            model,
            age=35,
            workclass='Private',
            education='Bachelors',
            education_num=13,
            hours_per_week=45,
            capital_gain=5000,
            occupation='Prof-specialty',
            marital_status='Married-civ-spouse',
            sex='Male',
            race='White'
        )
        print(f"    → {'>50K' if pred == 1 else '<=50K'} (probability: {proba:.2%})")
        
        # Example 3: Low earner
        print("\n  Example 3 (Low earner):")
        print("  Age: 22, Education: HS-grad, Hours: 20, Capital Gain: 0")
        pred, proba = predict_single(
            model,
            age=22,
            workclass='Private',
            education='HS-grad',
            education_num=10,
            hours_per_week=20,
            capital_gain=0,
            occupation='Other-service',
            marital_status='Never-married',
            sex='Female',
            race='Black'
        )
        print(f"    → {'>50K' if pred == 1 else '<=50K'} (probability: {proba:.2%})")
        
        # Example 4: Self-employed
        print("\n  Example 4 (Self-employed):")
        print("  Age: 45, Education: Masters, Hours: 50, Capital Gain: 0")
        pred, proba = predict_single(
            model,
            age=45,
            workclass='Self-emp-not-inc',
            education='Masters',
            education_num=14,
            hours_per_week=50,
            capital_gain=0,
            occupation='Exec-managerial',
            marital_status='Married-civ-spouse',
            sex='Male',
            race='White'
        )
        print(f"    → {'>50K' if pred == 1 else '<=50K'} (probability: {proba:.2%})")
        
        print("\n💡 To predict on a CSV file:")
        print("  python inference.py --input data/people.csv --threshold 0.4")

if __name__ == "__main__":
    main()