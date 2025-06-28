
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gzip
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="Prédicteur de Baccarat",
    page_icon="🎰",
    layout="wide"
)

# Titre
st.title('🎯 Prédictions Baccarat - 1440 Parties/Jour')
st.markdown("Prédictions pour la journée complète du lendemain")

# Chargement des modèles
@st.cache_resource
def load_models():
    try:
        # Charger le modèle compressé depuis le même répertoire
        with gzip.open('baccarat_model.pkl.gz', 'rb') as f:
            models, encoders = joblib.load(f)
        return models, encoders, True
    except Exception as e:
        st.error(f"Erreur de chargement des modèles: {str(e)}")
        return None, None, False

models, encoders, models_loaded = load_models()

# Statut des modèles - CORRECTION D'INDENTATION ICI
if models_loaded:
    st.success("✅ Modèles chargés avec succès!")
else:
    st.warning("⚠️ Mode simulation activé (modèles non chargés)")

# Formulaire de date
tomorrow = datetime.now() + timedelta(days=1)
date = st.date_input("Sélectionnez la date", tomorrow)
st.markdown(f"### Prédictions pour le {date.strftime('%d/%m/%Y')}")

# Calcul des prédictions
def generate_predictions(date):
    predictions = []
    total_minutes = 24 * 60  # 1440 minutes
    
    # Caractéristiques moyennes
    avg_features = [5.5, 5.5, 2.5, 2.5]
    
    for minute in range(total_minutes):
        hour = minute // 60
        min = minute % 60
        game_id = f"{date.strftime('%Y%m%d')}-{hour:02d}{min:02d}"
        
        if models and encoders:
            pred = {
                'ID Partie': game_id,
                'Heure': f"{hour:02d}:{min:02d}",
                '3ème Carte Joueur': models['Joueur_ThirdCard'].predict([avg_features])[0],
                '3ème Carte Banquier': models['Banquier_ThirdCard'].predict([avg_features])[0],
                'Victoire Joueur': models['Player_Win'].predict([avg_features])[0],
                'Victoire Banquier': models['Banker_Win'].predict([avg_features])[0]
            }
            
            # Prédiction des couleurs
            if pred['3ème Carte Joueur']:
                suit_code = models['Joueur_Suit'].predict([avg_features])[0]
                pred['Couleur Joueur'] = encoders['Joueur_Suit'].inverse_transform([suit_code])[0]
            else:
                pred['Couleur Joueur'] = 'Aucune'
                
            if pred['3ème Carte Banquier']:
                suit_code = models['Banquier_Suit'].predict([avg_features])[0]
                pred['Couleur Banquier'] = encoders['Banquier_Suit'].inverse_transform([suit_code])[0]
            else:
                pred['Couleur Banquier'] = 'Aucune'
        else:
            # Mode simulation
            pred = {
                'ID Partie': game_id,
                'Heure': f"{hour:02d}:{min:02d}",
                '3ème Carte Joueur': np.random.choice([0, 1], p=[0.7, 0.3]),
                '3ème Carte Banquier': np.random.choice([0, 1], p=[0.6, 0.4]),
                'Victoire Joueur': np.random.choice([0, 1], p=[0.45, 0.55]),
                'Victoire Banquier': np.random.choice([0, 1], p=[0.5, 0.5]),
                'Couleur Joueur': np.random.choice(['♣️ Pique', '♦️ Carreau', '♥️ Cœur', '♠️ Trèfle']),
                'Couleur Banquier': np.random.choice(['♣️ Pique', '♦️ Carreau', '♥️ Cœur', '♠️ Trèfle'])
            }
        
        predictions.append(pred)
    
    return pd.DataFrame(predictions)

# Bouton de génération
if st.button('🔮 Générer les prédictions'):
    with st.spinner('Calcul en cours... Cette opération peut prendre quelques minutes'):
        predictions_df = generate_predictions(date)
    
    # Afficher un aperçu
    st.dataframe(predictions_df.head(10))
    
    # Bouton de téléchargement
    csv = predictions_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger toutes les prédictions (CSV)",
        data=csv,
        file_name=f"baccarat_predictions_{date.strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
    
    # Visualisation
    st.markdown("### 📊 Tendances des prédictions")
    hourly_wins = predictions_df.groupby(predictions_df['Heure'].str[:2]).agg({
        'Victoire Joueur': 'mean',
        'Victoire Banquier': 'mean'
    }).reset_index()
    
    st.line_chart(hourly_wins.set_index('Heure'))
    
    # Statistiques
    col1, col2 = st.columns(2)
    with col1:
        player_wins = predictions_df['Victoire Joueur'].mean()
        st.metric("Victoires moyennes Joueur", f"{player_wins:.2%}")
    
    with col2:
        banker_wins = predictions_df['Victoire Banquier'].mean()
        st.metric("Victoires moyennes Banquier", f"{banker_wins:.2%}")

# Pied de page
st.markdown("---")
