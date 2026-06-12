const express = require('express');
const uploadRoutes = require('./routes/upload');

const app = express();

app.use('/api', uploadRoutes);

app.listen(8080, () => {
    console.log('Server is running on port 8080');
});