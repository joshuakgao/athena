# 121,322 games, 17.8MB
wget -P datasets/lichess/data https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
pzstd -d datasets/lichess/data/lichess_db_standard_rated_2013-01.pgn.zst
mv datasets/lichess/data/lichess_db_standard_rated_2013-01.pgn datasets/lichess/data/small.pgn
rm datasets/lichess/data/lichess_db_standard_rated_2013-01.pgn.zst

# # 1,000,056 games, 179MB
# wget -P datasets/lichess/data https://database.lichess.org/standard/lichess_db_standard_rated_2014-09.pgn.zst
# pzstd -d datasets/lichess/data/lichess_db_standard_rated_2014-09.pgn.zst
# mv datasets/lichess/data/lichess_db_standard_rated_2014-09.pgn datasets/lichess/data/medium.pgn
# rm datasets/lichess/data/lichess_db_standard_rated_2014-09.pgn.zst

# # 10,194,939 games, 1.8GB
# wget -P datasets/lichess/data https://database.lichess.org/standard/lichess_db_standard_rated_2017-02.pgn.zst
# pzstd -d datasets/lichess/data/lichess_db_standard_rated_2017-02.pgn.zst
# mv datasets/lichess/data/lichess_db_standard_rated_2017-02.pgn datasets/lichess/data/large.pgn
# rm datasets/lichess/data/lichess_db_standard_rated_2017-02.pgn.zst

# # 100,023,791 games, 32.4GB
# wget -P datasets/lichess/data https://database.lichess.org/standard/lichess_db_standard_rated_2021-03.pgn.zst
# pzstd -d datasets/lichess/data/lichess_db_standard_rated_2021-03.pgn.zst
# mv datasets/lichess/data/lichess_db_standard_rated_2021-03.pgn datasets/lichess/data/giant.pgn
# rm datasets/lichess/data/lichess_db_standard_rated_2021-03.pgn.zst