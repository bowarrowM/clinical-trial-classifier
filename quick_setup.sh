#!/bin/bash

echo " one command setup ver timeline "
echo "============================================================"

echo ""
echo "📊 Step 1/5: Generate synthetic dataset"
python synthetic_data.py

echo ""
echo "Step 2/5: Preprocess data"
python data_preprocess.py

echo ""
echo "Step 3/5: Train model (this may take some time)"
python model.py

echo ""
echo "Step 4/5: Evaluate model"
python model_evaluate.py

echo ""
echo "Step 5/5: Test LLM reasoning"
python llm_reasonings.py

echo ""
echo "============================================================"
echo "ALL STEPS COMPLETE"
echo ""
echo "To start the API server, run:"
echo "  python app.py"
echo ""
echo "To test the API in another terminal:"
echo "  python api_testing.py"
echo "============================================================"