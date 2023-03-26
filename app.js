const express = require('express');
const app = express();
const path = require('path');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');
const http = require('http');
const { resolve } = require('path');
const { promisify } = require('util');
const readFile = promisify(fs.readFile);
var hasran
app.use(cors());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(__dirname + '/public'));

app.listen(3000, () => console.log('Visit http://127.0.0.1:3000'));

let globalResponse;
const PythonShell = require('python-shell').PythonShell;

// Watch the 'trained' file for changes
PythonShell.run('python_code/main.py');

app.post('/write_to_json', (req, res) => {  
  globalResponse = res;
  console.log('Received POST request');
  const roomData = req.body;
  console.log(`Received room data: ${JSON.stringify(roomData)}`);

  // Write the data to individual JSON files for each room
  roomData.rooms.forEach((room, index) => {
    const fileName = `./public/json/gh${index + 1}_settings.json`;
    const dataObj = {
      minTemp: room.minTemp,
      maxTemp: room.maxTemp,
      rateOfChange: room.rateOfChange,
      critMinTemp: room.critMinTemp,
      critMaxTemp: room.critMaxTemp,
      maxTime: room.maxTime
    };
    const dataStr = JSON.stringify(dataObj, null, 2);
    fs.writeFile(fileName, dataStr, (err) => {
      if (err) {
        console.error(err);
        res.status(500).send(`Error writing data to file ${fileName}!`);
      } else {
        console.log(`Data written to file ${fileName} successfully!`);
      }
    });
  });
});
