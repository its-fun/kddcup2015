cd data

mongoimport -d xuetangx -c object --file=object.csv --type=csv --headerline

mongoimport -d xuetangx -c enrollment_test --file=test/enrollment_test.csv --type=csv --headerline
mongoimport -d xuetangx -c log_test --file=test/log_test.csv --type=csv --headerline

mongoimport -d xuetangx -c enrollment_train --file=train/enrollment_train.csv --type=csv --headerline
mongoimport -d xuetangx -c log_train --file=train/log_train.csv --type=csv --headerline


mongodump -d xuetangx -c object

mongodump -d xuetangx -c enrollment_test
mongodump -d xuetangx -c log_test

mongodump -d xuetangx -c enrollment_train
mongodump -d xuetangx -c log_train


mongorestore -d xuetangx -c log_all dump/xuetangx/log_train.bson
mongorestore -d xuetangx -c log_all dump/xuetangx/log_test.bson

mongorestore -d xuetangx -c enroll_all dump/xuetangx/enrollment_train.bson
mongorestore -d xuetangx -c enroll_all dump/xuetangx/enrollment_test.bson
