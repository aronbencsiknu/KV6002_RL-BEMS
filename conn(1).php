<?php
DEFINE ('DB_USER','root');
DEFINE ('DB_PASSWORD','LAYcxd1219...');
DEFINE ('DB_HOST','localhost');
DEFINE ('DB_NAME','new_schema_Greenhouse');

$dbc=@mysqli_connect(DB_HOST,DB_USER,DB_PASSWORD,DB_NAME) OR die('Could not to connect to Mysql:'.mysqli_connect_error());

mysqli_set_charset($dbc, 'utf8');
?>