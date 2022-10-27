var express = require('express');
const { recaptchaSecret } = require("../secret/recaptcha");
const axios = require('axios').default;
const url = require('url');
var router = express.Router();
const db = require('../database/db');

/* GET /user 
  Renders sign in form.
*/
router.get('/', function (req, res, next) {
  res.render("user-auth");
});

/* POST /user
  Processes user sign up form.
*/
router.post('/', async function (req, res, next) {
  const email = req.body.inputMail;
  const ageGroup = req.body.ageGroup;
  const education = req.body.education;
  const mlExpert = req.body.mlExpert;

  // Check reacaptcha
  const params = new url.URLSearchParams({ secret: recaptchaSecret, response: req.body["g-recaptcha-response"] });
  const response = await axios.post("https://www.google.com/recaptcha/api/siteverify", params);

  // Filter requests with missing recaptcha
  if (!response.data.success)
    res.render("user-auth", { error: "Incorrenct captcha!" });

  // Grad a user prepared from the database
  db.query("UPDATE users SET taken = true, email = $1, age_group = $2, education = $3, ml_expert = $4" +
    " where id = (select min(id) from users  where not taken ) RETURNING *", [email, ageGroup, education, mlExpert], (err, result) => {
      if (err) {
        console.log(err);
        res.status(500).send("Getting new users did not go well :)");
      }

      if (result.rows.length === 1) {
        // Render a form with user detail
        res.render("user", {
          user: result.rows[0], continueUrl: req.protocol + '://' + result.rows[0].name +
            ":" + result.rows[0].password + "@" + req.get('host') + "/annotation"
        });
      } else {
        res.status(404).send("No user was found.");
      }
    });
});

module.exports = router;
