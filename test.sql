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
`numofrooms` varchar(10) NOT NULL,
`location` varchar(200),
`code` varchar(10),
`message` text
) DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;