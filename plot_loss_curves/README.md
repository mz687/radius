To plot the loss curve, you need to copy and paste the logs to `./logs`.

After copying, modify `./run_scraper.sh` to scrape the loss and ppl per iteration and save the results to a csv file (e.g. `./logs/345M/dense/log_PP1_TP1_DP4.csv`).

Finally, add the paths to all the logs that you want to plot onto the figure. 