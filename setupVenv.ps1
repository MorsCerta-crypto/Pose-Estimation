# set variables for setup
$ModuleDirectory = "PoseClassifier"
$VenvName = "venv" # args[0]
$ActivateScript = ".\$VenvName\Scripts\Activate.ps1"

# setup python environment
Write-Host "Creating virtual environment for module $ModuleDirectory"
python -m venv $VenvName
Write-Host "Activating virtual environment"
& $ActivateScript
Write-Host "Installing requirements for module $ModuleDirectory"
pip install -r requirements.txt
Write-Host "Finished environment setup"

