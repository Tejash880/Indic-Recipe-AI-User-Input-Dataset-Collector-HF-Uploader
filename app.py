import os
import streamlit as st
from huggingface_hub import login, HfApi, HfFolder
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import hashlib

# Page configuration
st.set_page_config(
    page_title="🍛 Indic Recipe AI", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_FILE = "recipes.csv"
IMAGES_DIR = "recipe_images"
CONFIG_FILE = "app_config.json"

# Create directories if they don't exist
Path(IMAGES_DIR).mkdir(exist_ok=True)

# -------------------
# Utility Functions
# -------------------

@st.cache_data
def load_existing_data():
    """Load existing recipe data with caching"""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

def save_config(config):
    """Save app configuration"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)

def load_config():
    """Load app configuration"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def validate_recipe_data(name, ingredients, instructions):
    """Validate recipe input data"""
    errors = []
    if not name or len(name.strip()) < 3:
        errors.append("Recipe name must be at least 3 characters long")
    if not ingredients or len(ingredients.strip()) < 10:
        errors.append("Ingredients must be at least 10 characters long")
    if not instructions or len(instructions.strip()) < 20:
        errors.append("Instructions must be at least 20 characters long")
    return errors

def save_image(image_file, recipe_name):
    """Save uploaded image with unique filename"""
    if image_file:
        # Create unique filename using hash of content and name
        file_hash = hashlib.md5(image_file.getvalue()).hexdigest()[:8]
        safe_name = "".join(c for c in recipe_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        extension = image_file.name.split('.')[-1]
        filename = f"{safe_name}_{file_hash}.{extension}"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_file.getvalue())
        return filename
    return None

def save_recipe_locally(recipe_data):
    """Save recipe to local CSV file"""
    df_new = pd.DataFrame([recipe_data])
    
    if os.path.exists(DATA_FILE):
        df_existing = pd.read_csv(DATA_FILE)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    
    df.to_csv(DATA_FILE, index=False)
    # Clear cache to reload data
    load_existing_data.clear()

@st.cache_resource
def initialize_hf_api(token):
    """Initialize Hugging Face API with caching"""
    try:
        login(token=token)
        return HfApi(), True
    except Exception as e:
        return None, str(e)

def upload_to_huggingface(repo_id, token, include_images=False):
    """Upload dataset to Hugging Face with optional images"""
    if not os.path.exists(DATA_FILE):
        return False, "No local dataset found."
    
    api, login_result = initialize_hf_api(token)
    if not api:
        return False, f"Login failed: {login_result}"
    
    try:
        # Upload CSV file
        api.upload_file(
            path_or_fileobj=DATA_FILE,
            path_in_repo="recipes.csv",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
        
        # Upload images if requested and they exist
        if include_images and os.path.exists(IMAGES_DIR):
            image_files = list(Path(IMAGES_DIR).glob("*"))
            if image_files:
                for img_file in image_files:
                    api.upload_file(
                        path_or_fileobj=str(img_file),
                        path_in_repo=f"images/{img_file.name}",
                        repo_id=repo_id,
                        repo_type="dataset",
                        token=token
                    )
        
        return True, f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}"
    except Exception as e:
        return False, f"Upload failed: {str(e)}"

# -------------------
# Main App
# -------------------

st.title("🍛 Indic Recipe AI — Dataset Collector & HF Uploader")
st.caption("Collect and manage Indic recipe datasets with easy Hugging Face integration")

# Load configuration
config = load_config()

# Sidebar for configuration and stats
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Hugging Face token input
    hf_token_input = st.text_input(
        "Hugging Face Token", 
        type="password", 
        help="Your HF token for dataset uploads",
        value=config.get('hf_token', '')
    )
    
    if hf_token_input:
        config['hf_token'] = hf_token_input
        save_config(config)
        
        api, login_result = initialize_hf_api(hf_token_input)
        if api:
            st.success("✅ HF Login successful!")
        else:
            st.error(f"❌ Login failed: {login_result}")
    
    st.divider()
    
    # Dataset statistics
    st.header("📊 Dataset Stats")
    df_existing = load_existing_data()
    if not df_existing.empty:
        st.metric("Total Recipes", len(df_existing))
        if 'language' in df_existing.columns:
            lang_counts = df_existing['language'].value_counts()
            st.write("**By Language:**")
            for lang, count in lang_counts.items():
                st.write(f"• {lang}: {count}")
    else:
        st.info("No recipes yet")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Add New Recipe")
    
    with st.form("recipe_form", clear_on_submit=True):
        recipe_name = st.text_input(
            "Recipe Name*", 
            placeholder="e.g., Hyderabadi Biryani"
        )
        
        col_lang, col_region = st.columns(2)
        with col_lang:
            language = st.selectbox(
                "Language*", 
                ["Telugu", "Hindi", "Tamil", "Kannada", "Malayalam", 
                 "Marathi", "Bengali", "Gujarati", "Punjabi", "Other"]
            )
        
        with col_region:
            region = st.text_input(
                "Region/State", 
                placeholder="e.g., Andhra Pradesh"
            )
        
        ingredients = st.text_area(
            "Ingredients*", 
            placeholder="List ingredients (one per line or comma-separated)",
            height=100
        )
        
        instructions = st.text_area(
            "Cooking Instructions*", 
            placeholder="Detailed step-by-step cooking instructions",
            height=150
        )
        
        col_time, col_servings = st.columns(2)
        with col_time:
            prep_time = st.number_input("Prep Time (minutes)", min_value=0, value=0)
            cook_time = st.number_input("Cook Time (minutes)", min_value=0, value=0)
        
        with col_servings:
            servings = st.number_input("Servings", min_value=1, value=4)
            difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        
        image_file = st.file_uploader(
            "Recipe Image (optional)", 
            type=["jpg", "jpeg", "png"],
            help="Upload an image of the finished dish"
        )
        
        submitted = st.form_submit_button("💾 Save Recipe", type="primary")
        
        if submitted:
            # Validate input
            errors = validate_recipe_data(recipe_name, ingredients, instructions)
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Save image if provided
                image_filename = save_image(image_file, recipe_name) if image_file else None
                
                # Prepare recipe data
                recipe_data = {
                    "name": recipe_name.strip(),
                    "language": language,
                    "region": region.strip() if region else "",
                    "ingredients": ingredients.strip(),
                    "instructions": instructions.strip(),
                    "prep_time_minutes": prep_time,
                    "cook_time_minutes": cook_time,
                    "total_time_minutes": prep_time + cook_time,
                    "servings": servings,
                    "difficulty": difficulty,
                    "image_filename": image_filename,
                    "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save locally
                save_recipe_locally(recipe_data)
                st.success("✅ Recipe saved successfully!")
                st.rerun()

with col2:
    st.subheader("🚀 Dataset Management")
    
    # Display recent recipes
    df_existing = load_existing_data()
    if not df_existing.empty:
        st.write("**Recent Recipes:**")
        recent_recipes = df_existing.tail(5)[['name', 'language', 'date_added']]
        for _, row in recent_recipes.iterrows():
            st.write(f"• **{row['name']}** ({row['language']})")
    
    st.divider()
    
    # Hugging Face upload section
    st.write("**Upload to Hugging Face**")
    
    repo_id = st.text_input(
        "Dataset Repo ID", 
        placeholder="username/indic-recipes",
        value=config.get('repo_id', '')
    )
    
    if repo_id:
        config['repo_id'] = repo_id
        save_config(config)
    
    include_images = st.checkbox(
        "Include images", 
        value=True, 
        help="Upload recipe images along with the dataset"
    )
    
    if st.button("🔄 Upload to HF", type="primary"):
        if not hf_token_input:
            st.error("Please enter your HF token first")
        elif not repo_id:
            st.error("Please enter a repository ID")
        elif df_existing.empty:
            st.error("No recipes to upload")
        else:
            with st.spinner("Uploading dataset..."):
                success, message = upload_to_huggingface(repo_id, hf_token_input, include_images)
                if success:
                    st.success(message)
                else:
                    st.error(message)

# Data preview section
if not df_existing.empty:
    st.subheader("📋 Dataset Preview")
    
    # Filter options
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        lang_filter = st.multiselect(
            "Filter by Language", 
            options=df_existing['language'].unique(),
            default=[]
        )
    
    # Apply filters
    filtered_df = df_existing.copy()
    if lang_filter:
        filtered_df = filtered_df[filtered_df['language'].isin(lang_filter)]
    
    # Display data
    st.dataframe(
        filtered_df[['name', 'language', 'region', 'difficulty', 'total_time_minutes', 'date_added']],
        use_container_width=True,
        hide_index=True
    )
    
    # Download option
    csv_data = df_existing.to_csv(index=False)
    st.download_button(
        "📥 Download Dataset (CSV)",
        data=csv_data,
        file_name=f"indic_recipes_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )