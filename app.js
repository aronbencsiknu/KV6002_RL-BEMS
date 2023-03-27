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

const { spawn } = require('child_process');

const pyScripts = [];

const useDatabase = false;

pyScripts[0] = spawn('python', ['python_code/main.py', '-g', 1]);

app.post('/write_to_json', (req, res) => { 
  const roomData = req.body;

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
    // database code below
    if (useDatabase){
      pool.query('SELECT * FROM user WHERE roomNumber = ?', [index + 1], (error, results, fields) => {
        if (error) {
          console.error('Error executing query: ', error);
          return;
        }
        if (results.length === 0) {
          // Room number does not exist, so insert a new row
          pool.query('INSERT INTO user (roomNumber, minTemp, maxTemp, rateOfChange, critMinTemp, critMaxTemp, maxTime) VALUES (?, ?, ?, ?, ?, ?, ?)', [index + 1, room.minTemp, room.maxTemp, room.rateOfChange, room.critMinTemp, room.critMaxTemp, room.maxTime], (error, results, fields) => {
            if (error) {
              console.error('Error executing query: ', error);
              return;
            }
            console.log('Data inserted successfully!');
          });
        } else {
          // Room number already exists, so update the existing row
          pool.query('UPDATE user SET minTemp = ?, maxTemp = ?, rateOfChange = ?, critMinTemp = ?, critMaxTemp = ?, maxTime = ? WHERE roomNumber = ?', [room.minTemp, room.maxTemp, room.rateOfChange, room.critMinTemp, room.critMaxTemp, room.maxTime, index + 1], (error, results, fields) => {
            if (error) {
              console.error('Error executing query: ', error);
              return;
            }
            console.log('Data updated successfully!');
          });
        }
   
      });
    }
    
 
  });

  current_num = parseInt(roomData.greenhouse_nums[0].current_num);
  prev_num = parseInt(roomData.greenhouse_nums[0].prev_num);
  //pyScripts[1] = spawn('python', ['python_code/main.py',"2"]);

  if (current_num > prev_num){
    //create processes
    
    for (let i = prev_num; i < current_num; i++) {
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
      let initialData = {
        minTemp: 20,
        maxTemp: 25,
        rateOfChange: 1,
        critMinTemp: 17,
        critMaxTemp: 30,
        maxTime: 60
      };

      // database code below
      if(useDatabase){
        pool.query('SELECT * FROM user WHERE roomNumber = ?', [i + 1], (error, results, fields) => {
          if (error) {
            console.error('Error executing query: ', error);
            return;
          }
          if (results.length === 0) {
            // Room number does not exist, so insert a new row
            pool.query('INSERT INTO user (roomNumber, minTemp, maxTemp, rateOfChange, critMinTemp, critMaxTemp, maxTime) VALUES (?, ?, ?, ?, ?, ?, ?)', [i + 1, initialData.minTemp, initialData.maxTemp, initialData.rateOfChange, initialData.critMinTemp, initialData.critMaxTemp, initialData.maxTime], (error, results, fields) => {
              if (error) {
                console.error('Error executing query: ', error);
                return;
              }
              console.log('Data inserted successfully!');
            });
          } else {
            // Room number already exists, so update the existing row
            pool.query('UPDATE user SET minTemp = ?, maxTemp = ?, rateOfChange = ?, critMinTemp = ?, critMaxTemp = ?, maxTime = ? WHERE roomNumber = ?', [initialData.minTemp, initialData.maxTemp, initialData.rateOfChange, initialData.critMinTemp, initialData.critMaxTemp, initialData.maxTime, i + 1], (error, results, fields) => {
              if (error) {
                console.error('Error executing query: ', error);
                return;
              }
              console.log('Data updated successfully!');
            });
          }
     
        });
      }
      
      var temp = i+1;
      pyScripts.push(spawn('python', ['python_code/main.py', '-g', i+1]));
    
    } 
    
  }
  else if(current_num < prev_num){
    //kill processes
    for (let i = prev_num-1; i >= current_num; i--) {
      pyScripts[i].kill();
    } 
  }
  res.status(200).send("done with file");


});


// database code below
if(useDatabase){
  const mysql = require('mysql');
const pool = mysql.createPool({
  connectionLimit: 10,
  host: 'localhost',
  user: 'root',
  password: '1234', 
  database: 'greenhouse',
});

pool.getConnection((err, connection) => {
  if (err) {
    console.error('Error connecting to database: ', err);
    return;
  }
  console.log('Connected to database!');

 
    pool.query('SELECT * FROM user', (error, results, fields,res) => {
      if (error) {
        console.error('Error executing query: ', error);
        res.send('Error executing query!');

      }
    console.log(fields);

      connection.release(); // release the connection back to the pool
    });
  });
﻿
}
