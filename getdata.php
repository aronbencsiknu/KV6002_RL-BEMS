<?php
//Connect to MySQL database
$servername = "localhost";
$username = "root";
$password = "LAYcxd1219...";
$dbname = "Greenhouse";
$conn = new mysqli($servername, $username, $password, $dbname);

//Check if the connection was successful
if ($conn->connect_error) {
    die("connection failed: " . $conn->connect_error);
}

// Read the data.json、room1.json、room2.json、room3.json、room4.json、room5.json、room6.json、room7.json、room8.json、Storage.json file

for ($i = 1; $i <= 8; $i++) {
    $filename = "room" . $i . ".json";
    $jsondata = file_get_contents($filename);
    $data = json_decode($jsondata, true);

    // Loop through room json data and insert it into the MySQL database
    //foreach ($data as $row) {
        $sql = "INSERT INTO user(roomNumber,minTemp,maxTemp,rateOfChange,critMinTemp,critMaxTemp,maxTime) VALUES (" . $i . ", '" . $data['minTemp'] . "', '" . $data['maxTemp'] . "', '" . $data['rateOfChange'] . "', '" . $data['critMinTemp'] . "', '" . $data['critMaxTemp'] . "', '" . $data['maxTime'] . "')";

if ($conn->query($sql) === TRUE) {
    echo "Data inserted successfully";
} else {
    echo "Error: " . $sql . "<br>" . $conn->error;
}
    }
//}
// Loop through the Storage data and insert it into the MySQL database
$storagedata = file_get_contents('Storage.json');
$data1 = json_decode($storagedata, true);
$numofrooms =$data1['numofrooms'];
$location = $data1['location'];
$forecast_cod =$data1['forecast']['cod'];
$forecast_message =$data1['forecast']['message'];
$sql1 = "INSERT INTO historical(numofrooms,location,code,message) VALUES ('$numofrooms', '$location', '$forecast_cod', '$forecast_message')";

if ($conn->query($sql1) === TRUE) {
    echo "Data inserted successfully";
} else {
    echo "Error: " . $sql1 . "<br>" . $conn->error;
}

$conn->close();
?>