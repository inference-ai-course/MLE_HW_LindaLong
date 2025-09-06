#!/bin/bash

pip3 install -U langchain-community
pip3 install pypdf
pip3 install --upgrade langchain langchain-core langchain-community langchain-openai langchain-experimental chromadb
pip3 install openai
pip3 install faiss-cpu
pip3 install sentence-transformers
pip3 install PyMuPDF

#make the script executable and run it:
# chmod +x install_class4_requirements.sh
#./install_class4_requirements.sh