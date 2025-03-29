const { exec } = require('child_process');

exports.checkFoodSuitability = (req, res) => {
    const { ingredients } = req.body;

    exec(`python3 ai_model/predict.py "${ingredients}"`, (error, stdout, stderr) => {
        if (error) {
            return res.status(500).json({ error: "Error processing request" });
        }
        res.json(JSON.parse(stdout));
    });
};
