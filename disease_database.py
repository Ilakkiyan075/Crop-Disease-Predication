"""
Disease database containing information about crop diseases and their remedies.
"""

DISEASE_DATABASE = {
    'apple_scab': {
        'name': 'Apple Scab',
        'description': 'A fungal disease that affects apple trees, causing dark spots on leaves and fruit.',
        'symptoms': [
            'Dark, olive-green to black spots on leaves',
            'Spots on fruit surface',
            'Premature leaf drop',
            'Reduced fruit quality'
        ],
        'remedies': [
            'Apply fungicide sprays during spring',
            'Remove fallen leaves and debris',
            'Prune trees for better air circulation',
            'Plant resistant apple varieties',
            'Use copper-based fungicides'
        ],
        'prevention': [
            'Regular pruning',
            'Proper spacing between trees',
            'Avoid overhead watering',
            'Clean up fallen leaves'
        ]
    },
    'bacterial_blight': {
        'name': 'Bacterial Blight',
        'description': 'A bacterial infection affecting various crops, causing leaf spots and wilting.',
        'symptoms': [
            'Water-soaked spots on leaves',
            'Yellow halos around spots',
            'Wilting of affected parts',
            'Stem cankers'
        ],
        'remedies': [
            'Apply copper-based bactericides',
            'Remove infected plant parts',
            'Improve drainage',
            'Use resistant varieties',
            'Avoid overhead irrigation'
        ],
        'prevention': [
            'Crop rotation',
            'Proper field sanitation',
            'Use certified disease-free seeds',
            'Avoid working in wet conditions'
        ]
    },
    'powdery_mildew': {
        'name': 'Powdery Mildew',
        'description': 'A fungal disease creating white powdery coating on plant surfaces.',
        'symptoms': [
            'White powdery coating on leaves',
            'Yellowing of leaves',
            'Stunted growth',
            'Distorted leaves and shoots'
        ],
        'remedies': [
            'Apply sulfur-based fungicides',
            'Use baking soda spray (1 tsp per quart water)',
            'Improve air circulation',
            'Remove affected plant parts',
            'Apply neem oil'
        ],
        'prevention': [
            'Plant in sunny locations',
            'Ensure proper spacing',
            'Avoid overhead watering',
            'Choose resistant varieties'
        ]
    },
    'leaf_spot': {
        'name': 'Leaf Spot',
        'description': 'Various fungal or bacterial diseases causing spots on leaves.',
        'symptoms': [
            'Circular or irregular spots on leaves',
            'Brown, black, or yellow spots',
            'Premature leaf drop',
            'Reduced photosynthesis'
        ],
        'remedies': [
            'Apply appropriate fungicides',
            'Remove infected leaves',
            'Improve air circulation',
            'Avoid overhead watering',
            'Use copper-based sprays'
        ],
        'prevention': [
            'Proper plant spacing',
            'Water at soil level',
            'Remove plant debris',
            'Rotate crops annually'
        ]
    },
    'rust': {
        'name': 'Rust Disease',
        'description': 'Fungal diseases causing rust-colored spots on plant surfaces.',
        'symptoms': [
            'Orange, yellow, or brown pustules',
            'Rust-colored spots on leaves',
            'Premature leaf drop',
            'Weakened plant growth'
        ],
        'remedies': [
            'Apply fungicides containing propiconazole',
            'Remove infected plant parts',
            'Improve air circulation',
            'Use resistant varieties',
            'Apply preventive sprays'
        ],
        'prevention': [
            'Plant resistant varieties',
            'Ensure good drainage',
            'Avoid overhead irrigation',
            'Remove alternate hosts'
        ]
    },
    'healthy': {
        'name': 'Healthy Plant',
        'description': 'No disease detected. Plant appears healthy.',
        'symptoms': [
            'Green, vibrant foliage',
            'Normal growth pattern',
            'No visible spots or discoloration',
            'Strong stem and root system'
        ],
        'remedies': [
            'Continue current care practices',
            'Maintain proper watering schedule',
            'Ensure adequate nutrition',
            'Monitor for early disease signs'
        ],
        'prevention': [
            'Regular monitoring',
            'Proper fertilization',
            'Adequate water management',
            'Good sanitation practices'
        ]
    }
}

def get_disease_info(disease_name):
    """
    Get detailed information about a specific disease.
    
    Args:
        disease_name (str): Name of the disease
        
    Returns:
        dict: Disease information including symptoms, remedies, and prevention
    """
    return DISEASE_DATABASE.get(disease_name.lower(), {
        'name': 'Unknown Disease',
        'description': 'Disease not found in database.',
        'symptoms': ['Unable to identify specific symptoms'],
        'remedies': ['Consult with agricultural extension services', 'Contact local plant pathologist'],
        'prevention': ['Follow general plant health practices']
    })

def get_all_diseases():
    """
    Get list of all diseases in the database.
    
    Returns:
        list: List of disease names
    """
    return list(DISEASE_DATABASE.keys())
