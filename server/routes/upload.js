const express = require('express');
const multer = require('multer');
const {exec} = require('child_process');

const router = express.Router();

const upload = multer({dest: 'uploads/'});

router.post("/upload", upload.single("image"),(req, res) => {
    if(!req.file){
        return res.status(400).json({error: "No file uploaded"});
    }

    const imagePath = req.file.path;

    exec(
    //    `python ai_worker/ocr_only.py ${imagePath}`,
        `python ai_worker/paddle_ocr.py "${imagePath}"`,
        (error, stdout, stderr) => {
            if(error){
                return res.status(500).json({error: stderr || error.message});
            }

            res.json({
                message: "OCR extraction succesful",
                extracted_text : stdout
            });
        }
    )
})

module.exports = router;