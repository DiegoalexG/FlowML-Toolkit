# Dictionary to define the name, window class location, and parameters related to each tool
tools_dict = {
    "vof_prediction": {
        "name": "VOF Prediction",
        "description": "VOF Prediction is a machine learning application designed to predict the spatio-temporal evolution of viscoelastic droplets " \
                       "impacting solid surfaces. The tool uses deep learning architectures, including Video Vision Transformers (ViViT), to forecast " \
                       "future volume fraction fields from an initial simulation context, providing a faster alternative to full computational fluid " \
                       "dynamics simulations while preserving important physical behaviors such as spreading and bouncing dynamics. \n" \
                       "The application is based on datasets generated with the Basilisk flow solver using the Volume of Fluid (VOF) method, " \
                       "considering variations in Reynolds number (Re), Weber number (We), solvent viscosity ratio (β), and Weissenberg number (Wi). " \
                       "In addition to visualizing predicted droplet evolution, the tool also evaluates geometric and image-quality metrics, including " \
                       "droplet diameters, center of mass evolution, contact time, R²-score, RMSE, and SSIM, enabling detailed comparison between predicted " \
                       "and reference simulations.",
        "view": "tools.vof_prediction.view.VOFPredictionView",
        "params": {
            "methods": "tools.vof_prediction.methods.VOFPrediction",
            "architectures": "tools.vof_prediction.architectures.architecture",
            "examples": r"tools\vof_prediction\examples",
            "models": {
                "Model 50→1": r"tools\vof_prediction\models\model_vivit_50_1.keras", 
                "Model 50→50": r"tools\vof_prediction\models\model_vivit_50_50.keras", 
                "Model 100→100": r"tools\vof_prediction\models\model_vivit_100_100.keras"
            }
        }
    }
}