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

app.use(cors());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static(__dirname + '/public'));

app.listen(3000, () => console.log('Visit http://localhost:3000/'));

//const PythonShell = require('python-shell').PythonShell;
const { spawn } = require('child_process');

/*
const runScript = (args) => {
  const python = spawn('python', ['python_code/main.py',args]);
}
runScript(["2"]);*/

// Watch the 'trained' file for changes
//PythonShell.run('python_code/main.py');
const pyScripts = [];

pyScripts[0] = spawn('python', ['python_code/main.py', '-g', 1]);

app.post('/write_to_json', (req, res) => { 
  const roomData = req.body;
  //PythonShell.run('python_code/main.py');
  // Write the data to individual JSON files for each room
  var dataStr; 
  
  roomData.rooms.forEach((room, index) => {
    const fileName = `./public/json/gh${index + 1}_settings.json`;
    
    let dataObj = {
      minTemp: room.minTemp,
      maxTemp: room.maxTemp,
      rateOfChange: room.rateOfChange,
      critMinTemp: room.critMinTemp,
      critMaxTemp: room.critMaxTemp,
      maxTime: room.maxTime
    };
    dataStr = JSON.stringify(dataObj, null, 2);
    fs.writeFileSync(fileName, dataStr, (err) => {
      if (err) {
        console.error(err);
        res.status(500).send(`Error writing data to file ${fileName}!`);
      } else {
  
      }
    });
  });
  current_num = parseInt(roomData.greenhouse_nums[0].current_num);
  prev_num = parseInt(roomData.greenhouse_nums[0].prev_num);
  //pyScripts[1] = spawn('python', ['python_code/main.py',"2"]);

  if (current_num > prev_num){
    //create processes
    
    for (let i = prev_num; i <= current_num; i++) {
      const fileName = `./public/json/gh${i}_settings.json`;
      if (!fs.existsSync(fileName)) {
        // If it doesn't exist, create it with some initial data
        let initialData = {
          minTemp: 20,
          maxTemp: 25,
          rateOfChange: 1,
          critMinTemp: 17,
          critMaxTemp: 30,
          maxTime: 60
        };
        fs.writeFileSync(fileName, JSON.stringify(initialData));
      }
      var temp = i+1;
      pyScripts.push(spawn('python', ['python_code/main.py', '-g', i+1]));
    } 
    
  }
  else if(current_num < prev_num){
    //kill processes
    for (let i = prev_num; i < current_num; i--) {
      pyScripts[i].kill();
    } 
  }
  
  res.status(200).send("done with file");

});
