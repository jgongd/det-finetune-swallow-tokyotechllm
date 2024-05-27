
/usr/bin/python -m pip install --upgrade pip
pip install ipywidgets widgetsnbextension
pip uninstall flash-attn -y
pip -q install transformers==4.34.0 #downgrade
pip uninstall transformer-engine -y
pip install peft==0.10.0 