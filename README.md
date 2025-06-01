# Projet SMART - Smart Merchandise Automated Recognition Technology

Système de détection d'objets utilisant YOLO pour reconnaître automatiquement 10 produits spécifiques.

## 🚀 Installation

### Prérequis
- Python 3.11
- GPU NVIDIA avec CUDA (recommandé pour les performances)
- Compte Picsellia pour accéder au dataset
- Compte BentoML Cloud pour le déploiement

### Installation des dépendances
```bash
# Cloner le repository
git clone https://github.com/votre-username/projet-smart.git
cd projet-smart

# Installer les dépendances
pip install -r requirements.txt

# Installer pre-commit
pip install pre-commit
pre-commit install
```

## ⚙️ Configuration

### 1. Variables d'environnement
Copier le fichier `.env.example` et le renommer en `.env` :
```bash
cp src/config/.env.example src/config/.env
```

### 2. Démarrer les services MLflow
```bash
docker-compose up -d
```

### 3. Configurer BentoML Cloud
```bash
bentoml cloud login
```

## 📊 Utilisation

### 🎯 Pipeline Complète (Recommandée)

```bash
# Pipeline complète avec déploiement cloud
python src/main_pipeline.py

# Pour développement local uniquement
python src/main_pipeline.py --serve-locally
```

### 🔍 Pipeline d'Inférence

```bash
# Mode IMAGE
python src/inference_pipeline.py --mode IMAGE --input_path path/to/image.jpg

# Mode VIDEO
python src/inference_pipeline.py --mode VIDEO --input_path path/to/video.mp4

# Mode WEBCAM (pour démonstration)
python src/inference_pipeline.py --mode WEBCAM
```

## 🧪 Tests et Validation

```bash
# Tests unitaires
pytest

# Validation complète avec coverage
pytest --cov=src --cov-report=html

# Test du service déployé
python -c "import requests; print(requests.get('YOUR_BENTOML_URL/health').json())"
```

## 🏷️ Classes Détectées (10 produits)

1. **Canette**
2. **Compote** 
3. **Brownie** 
4. **Twix** 
5. **Kinder_cards** 
6. **Sondey_carré**
7. **Barre_protéine** 
8. **Mars**
9. **Oeuf**
10. **Lapin_pâques** 


## 🔧 MLOps & Bonnes Pratiques

### ✅ Fonctionnalités Implémentées
- **🔄 Pipeline ML complète** : Download → Prepare → Train → Deploy
- **📊 MLflow Integration** : Experiment tracking + Model Registry
- **🚀 BentoML Serving** : Déploiement cloud automatisé
- **🧪 Tests Automatisés** : Coverage + CI/CD
- **⚙️ Configuration Flexible** : CPU/GPU auto-détection
- **📱 Multi-modal Inference** : Image/Video/Webcam
- **🛡️ Error Handling** : Fallbacks robustes
- **📋 Code Quality** : Pre-commit hooks + Linting

### 🎯 Points d'Évaluation Couverts
- [x] Correcte utilisation d'Ultralytics, MLflow, BentoML, Picsellia
- [x] Qualité du code Python (type hints, docstrings, formatting)
- [x] Utilisation correcte de Picsellia pour les datasets
- [x] MLflow pour Metadata Store et Model Registry
- [x] BentoML pour le Serving cloud
- [x] 2 pipelines (training + inférence) fonctionnelles
- [x] README complet avec instructions
- [x] Commits conventionnels

## 🚀 Déploiement et Démonstration

### Service BentoML Cloud
- **URL** : [Votre URL BentoML Cloud]
- **Endpoints** :
  - `GET /health` : Status du service
  - `POST /predict` : Prédiction sur image

### Interface de Monitoring
- **MLflow** : http://localhost:5000
- **Métriques** : Tracking automatique des performances

## 👥 Équipe

- **Fouad Belhia**
- **Ghada Ben Younes**
- **Oumaima Moughazli**

## 📈 Améliorations Futures

- [ ] Data augmentation avancée
- [ ] Ensemble de modèles
- [ ] Monitoring en temps réel avec Grafana

---

**🎓 Projet réalisé dans le cadre du CNAM - Université de Strasbourg**  
**📚 Module : Intelligence Artificielle - MLOps**