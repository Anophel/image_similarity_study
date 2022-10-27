var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var logger = require('morgan');
const basicAuth = require('express-basic-auth');
var bodyParser = require('body-parser');
const db = require('./database/db');

var indexRouter = require('./routes/index');
var userRouter = require('./routes/user');
var explorerRouter = require('./routes/explorer');

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

// Set logging, data formats and cookie parser
app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

// Add user sign in form without authentication
app.use('/user', userRouter);

// Simple authorization function
function myAsyncAuthorizer(username, password, cb) {
  // Query if the user with the username and password exists
  db.query("select * from users where name = $1 and password = $2 and taken", [username, password], (err, result) => {
    if (err) {
      console.log(err);
      cb(null, false);
    }

    if (result.rows.length === 1) {
      cb(null, true);
    } else {
      cb(null, false);
    }
  });
}

// Wire public folder as static folder
app.use(express.static(path.join(__dirname, 'public')));

app.use(bodyParser.urlencoded({ extended: true }));

// Prepare basic auth object
const authentication = basicAuth({
  authorizer: myAsyncAuthorizer,
  authorizeAsync: true,
  challenge: true,
  realm: 'f40VOWEI0'
});

// Add annotation route with authentication
app.use('/annotation', authentication, indexRouter);
// Add explorer route with authentication
app.use('/explorer', authentication, explorerRouter);

// catch 404 and forward to error handler
app.use(function (req, res, next) {
  next(createError(404));
});

// error handler
app.use(function (err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
