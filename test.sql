--
-- Database : `Greenhouse`
--
CREATE DATABASE IF NOT EXISTS `Greenhouse`
  DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
USE `Greenhouse`;

-- --------------------------------------------------------
--
-- Table structure for table `user`
--
CREATE TABLE `user` (
`roomNumber` int(10),
`minTemp` varchar(20),
`maxTemp` varchar(20),
`rateOfChange` varchar(20), 
`critMinTemp` varchar(20), 
`critMaxTemp` varchar(20),
`maxTime` varchar(20)
) DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
--
-- Table structure for table `historical`
--

CREATE TABLE  `historical` (
`roomNumber` varchar(10) ,
`GREENHOUSE_TEMP` varchar(200),
`OUTSIDE_TEMP` varchar(10),

`HEAT` varchar(10),

`VENT` varchar(10),
`AVG_CONSUMPTION` varchar(10),
`Elapsed_time` varchar(10)

) DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;