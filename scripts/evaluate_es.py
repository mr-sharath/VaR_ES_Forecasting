# scripts/eval_es.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def evaluate_es():
    # Load data
    data = pd.read_csv('results/var_predictions/BLK.csv', index_col=0)
    test_returns = data['return'].values
    var_predictions = data['VaR_95'].values
    
    # Load generator
    generator = tf.keras.models.load_model('models/es_gan.h5')
    
    # Generate tail scenarios
    n_scenarios = 1000
    tail_mask = test_returns < var_predictions
    cond_vars = var_predictions[tail_mask]
    
    # Generate scenarios for each tail VaR
    scenarios = []
    for var in cond_vars:
        noise = np.random.normal(0, 1, (n_scenarios, 16))
        cond_input = np.column_stack([noise, np.full((n_scenarios, 1), var)])
        scenarios.extend(generator.predict(cond_input, verbose=0).flatten())
    
    # Calculate ES
    es_estimate = np.mean(scenarios)
    actual_tail_losses = test_returns[test_returns < var_predictions].mean()
    
    print(f"ES Evaluation Results:")
    print(f"Predicted ES: {es_estimate:.6f}")
    print(f"Actual Avg Tail Loss: {actual_tail_losses:.6f}")
    print(f"Absolute Error: {abs(es_estimate - actual_tail_losses):.6f}")

if __name__ == "__main__":
    evaluate_es()