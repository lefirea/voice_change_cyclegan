""" f0,spを分けて学習させるときのモデル定義 """
# if os.path.exists(self.log_path + "log.txt"):
#     """ 基本モデル読み込み """
#     self.Gen_A_f0 = model_from_json(open(self.log_path + "gen_a_f0_model.json", "r").read())
#     self.Gen_A_f0.compile(loss="binary_crossentropy",
#                           optimizer=self.gen_optimizer,
#                           # metrics=["accuracy"]
#                           )
#     self.Gen_A_f0.load_weights(self.log_path + "gen_a_f0_weights.h5")
#
#     self.Gen_A_sp = model_from_json(open(self.log_path + "gen_a_sp_model.json", "r").read())
#     self.Gen_A_f0.compile(loss="binary_crossentropy",
#                           optimizer=self.gen_optimizer,
#                           # metrics=["accuracy"]
#                           )
#     self.Gen_A_sp.load_weights(self.log_path + "gen_a_sp_weights.h5")
#
#     self.Disc_A_f0 = model_from_json(open(self.log_path + "disc_a_f0_model.json", "r").read())
#     self.Disc_A_f0.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#     self.Disc_A_f0.load_weights(self.log_path + "disc_a_f0_weights.h5")
#
#     self.Disc_A_sp = model_from_json(open(self.log_path + "disc_a_sp_model.json", "r").read())
#     self.Disc_A_sp.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#     self.Disc_A_sp.load_weights(self.log_path + "disc_a_sp_weights.h5")
#
#     self.Gen_B_f0 = model_from_json(open(self.log_path + "gen_b_f0_model.json", "r").read())
#     self.Gen_B_f0.compile(loss="binary_crossentropy",
#                           optimizer=self.gen_optimizer,
#                           # metrics=["accuracy"]
#                           )
#     self.Gen_B_f0.load_weights(self.log_path + "gen_b_f0_weights.h5")
#
#     self.Gen_B_sp = model_from_json(open(self.log_path + "gen_b_sp_model.json", "r").read())
#     self.Gen_B_sp.compile(loss="binary_crossentropy",
#                           optimizer=self.gen_optimizer,
#                           # metrics=["accuracy"]
#                           )
#     self.Gen_B_sp.load_weights(self.log_path + "gen_b_sp_weights.h5")
#
#     self.Disc_B_f0 = model_from_json(open(self.log_path + "disc_b_f0_model.json", "r").read())
#     self.Disc_B_f0.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#     self.Disc_B_f0.load_weights(self.log_path + "disc_b_f0_weights.h5")
#
#     self.Disc_B_sp = model_from_json(open(self.log_path + "disc_b_sp_model.json", "r").read())
#     self.Disc_B_sp.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#     self.Disc_B_sp.load_weights(self.log_path + "disc_b_sp_weights.h5")
#
#     """ 基本モデルから、残りのモデルを再構築 """
#     # モデルロード時に、各モデルの依存関係が失われる？未検証故によく分からぬ。
#     # 仮に失われてたときに備えて、再構築してウェイトを読み込む
#
#     self.combined_aB_f0 = self.combine(self.Gen_B_f0, self.Disc_B_f0)
#     self.combined_aB_f0.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"]
#                                 )
#     self.combined_aB_f0.load_weights(self.log_path + "combined_ab_f0_weights.h5")
#
#     self.combined_aB_sp = self.combine(self.Gen_B_sp, self.Disc_B_sp)
#     self.combined_aB_sp.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"]
#                                 )
#     self.combined_aB_sp.load_weights(self.log_path + "combined_ab_sp_weights.h5")
#
#     self.combined_bA_f0 = self.combine(self.Gen_A_f0, self.Disc_A_f0)
#     self.combined_bA_f0.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"]
#                                 )
#     self.combined_bA_f0.load_weights(self.log_path + "combined_ba_f0_weights.h5")
#
#     self.combined_bA_sp = self.combine(self.Gen_A_sp, self.Disc_A_sp)
#     self.combined_bA_sp.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"]
#                                 )
#     self.combined_bA_sp.load_weights(self.log_path + "combined_ba_sp_weights.h5")
#
#     self.Gen_aBA_f0 = Sequential([self.Gen_B_f0, self.Gen_A_f0])
#     self.Gen_aBA_f0.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer,
#                             # metrics=["accuracy"]
#                             )
#     self.Gen_aBA_f0.load_weights(self.log_path + "gen_aba_f0_weights.h5")
#
#     self.Gen_aBA_sp = Sequential([self.Gen_B_sp, self.Gen_A_sp])
#     self.Gen_aBA_sp.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer,
#                             # metrics=["accuracy"]
#                             )
#     self.Gen_aBA_sp.load_weights(self.log_path + "gen_aba_sp_weights.h5")
#
#     self.Gen_bAB_f0 = Sequential([self.Gen_A_f0, self.Gen_B_f0])
#     self.Gen_bAB_f0.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer,
#                             # metrics=["accuracy"]
#                             )
#     self.Gen_bAB_f0.load_weights(self.log_path + "gen_bab_f0_weights.h5")
#
#     self.Gen_bAB_sp = Sequential([self.Gen_A_sp, self.Gen_B_sp])
#     self.Gen_bAB_sp.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer,
#                             # metrics=["accuracy"]
#                             )
#     self.Gen_bAB_sp.load_weights(self.log_path + "gen_bab_sp_weights.h5")
#
# else:
#     """ モデル定義 """
#
#     """ 基本モデル """
#     # データA側
#     # f0ジェネレータ
#     self.Gen_A_f0 = self.gen.f0Generator(rate=10)
#     self.Gen_A_f0.compile(loss="binary_crossentropy",
#                           optimizer=not self.gen_optimizer)
#
#     # spジェネレータ
#     self.Gen_A_sp = self.gen.spGenerator(fft_len=1025)
#     self.Gen_A_sp.compile(loss="binary_crossentropy",
#                           optimizer=self.gen_optimizer)
#
#     # f0ディスクリミネータ
#     self.Disc_A_f0 = self.disc.f0Discriminator(rate=10)
#     self.Disc_A_f0.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#
#     # spディスクリミネータ
#     self.Disc_A_sp = self.disc.spDiscriminator(fft_len=1025)
#     self.Disc_A_sp.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#
#     # データB側
#     # f0ジェネレータ
#     self.Gen_B_f0 = self.gen.f0Generator(rate=10)
#     self.Gen_B_f0.compile(loss="binary_crossentropy",
#                           optimizer=not self.gen_optimizer)
#
#     # spジェネレータ
#     self.Gen_B_sp = self.gen.spGenerator(fft_len=1025)
#     self.Gen_B_sp.compile(loss="binary_crossentropy",
#                           optimizer=self.gen_optimizer)
#
#     # f0ディスクリミネータ
#     self.Disc_B_f0 = self.disc.f0Discriminator(rate=10)
#     self.Disc_B_f0.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#
#     # spディスクリミネータ
#     self.Disc_B_sp = self.disc.spDiscriminator(fft_len=1025)
#     self.Disc_B_sp.compile(loss="binary_crossentropy",
#                            optimizer=self.disc_optimizer,
#                            metrics=["accuracy"])
#
#     """" 基本モデルをもとに組み上げるモデル """
#     # データAをデータBに変換して真贋の評価を行うモデル
#     # Generatorだけ学習させる
#     self.combined_aB_f0 = self.combine(self.Gen_B_f0, self.Disc_B_f0)
#     self.combined_aB_f0.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"])
#     self.combined_aB_sp = self.combine(self.Gen_B_sp, self.Disc_B_sp)
#     self.combined_aB_sp.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"])
#
#     # データBをデータAに変換して真贋の評価を行うモデル
#     # Generatorだけ学習させる
#     self.combined_bA_f0 = self.combine(self.Gen_A_f0, self.Disc_A_f0)
#     self.combined_bA_f0.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"])
#
#     self.combined_bA_sp = self.combine(self.Gen_A_sp, self.Disc_A_sp)
#     self.combined_bA_sp.compile(loss="binary_crossentropy",
#                                 optimizer=self.gen_optimizer,
#                                 metrics=["accuracy"])
#
#     # データAをデータBに変換して、更にデータAに戻して、復元率を評価する
#     self.Gen_aBA_f0 = Sequential([self.Gen_B_f0, self.Gen_A_f0])
#     self.Gen_aBA_f0.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer)
#     self.Gen_aBA_sp = Sequential([self.Gen_B_sp, self.Gen_A_sp])
#     self.Gen_aBA_sp.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer)
#
#     # データBをデータAに変換して、更にデータBに戻して、復元率を評価する
#     self.Gen_bAB_f0 = Sequential([self.Gen_A_f0, self.Gen_B_f0])
#     self.Gen_bAB_f0.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer)
#     self.Gen_bAB_sp = Sequential([self.Gen_A_sp, self.Gen_B_sp])
#     self.Gen_bAB_sp.compile(loss="binary_crossentropy",
#                             optimizer=self.gen_optimizer)
#
#     """ モデル保存 """
#     # Gen_A_f0保存
#     Gen_A_f0_json = self.Gen_A_f0.to_json()
#     open(self.log_path + "gen_a_f0_model.json", "w").write(Gen_A_f0_json)
#     # Disc_A_f0保存
#     Disc_A_f0_json = self.Disc_A_f0.to_json()
#     open(self.log_path + "disc_a_model_f0.json", "w").write(Disc_A_f0_json)
#     # Gen_A_sp保存
#     Gen_A_sp_json = self.Gen_A_sp.to_json()
#     open(self.log_path + "gen_a_sp_model.json", "w").write(Gen_A_sp_json)
#     # Disc_A_sp保存
#     Disc_A_sp_json = self.Disc_A_sp.to_json()
#     open(self.log_path + "disc_a_model_sp.json", "w").write(Disc_A_sp_json)
#
#     # Gen_B_f0保存
#     Gen_B_f0_json = self.Gen_B_f0.to_json()
#     open(self.log_path + "gen_b_model_f0.json", "w").write(Gen_B_f0_json)
#     # Disc_B_f0保存
#     Disc_B_f0_json = self.Disc_B_f0.to_json()
#     open(self.log_path + "disc_b_model_f0.json", "w").write(Disc_B_f0_json)
#     # Gen_B_sp保存
#     Gen_B_sp_json = self.Gen_B_sp.to_json()
#     open(self.log_path + "gen_b_model_sp.json", "w").write(Gen_B_sp_json)
#     # Disc_B_sp保存
#     Disc_B_f0_json = self.Disc_B_f0.to_json()
#     open(self.log_path + "disc_b_model_sp.json", "w").write(Disc_B_f0_json)
#
#     # combined_aB_f0保存
#     combined_aB_f0_json = self.combined_aB_f0.to_json()
#     open(self.log_path + "combined_ab_f0_model.json", "w").write(combined_aB_f0_json)
#     combined_aB_sp_json = self.combined_aB_sp.to_json()
#     open(self.log_path + "combined_ab_sp_model.json", "w").write(combined_aB_sp_json)
#     # combined_bA保存
#     combined_bA_f0_json = self.combined_bA_f0.to_json()
#     open(self.log_path + "combined_ba_f0_model.json", "w").write(combined_bA_f0_json)
#     combined_bA_sp_json = self.combined_bA_sp.to_json()
#     open(self.log_path + "combined_ba_sp_model.json", "w").write(combined_bA_sp_json)
#
#     # Gen_aBA保存
#     Gen_aBA_f0_json = self.Gen_aBA_f0.to_json()
#     open(self.log_path + "gen_aba_f0_model.json", "w").write(Gen_aBA_f0_json)
#     Gen_aBA_sp_json = self.Gen_aBA_sp.to_json()
#     open(self.log_path + "gen_aba_sp_model.json", "w").write(Gen_aBA_sp_json)
#     # Gen_bAB保存
#     Gen_bAB_f0_json = self.Gen_bAB_f0.to_json()
#     open(self.log_path + "gen_bab_f0_model.json", "w").write(Gen_bAB_f0_json)
#     Gen_bAB_sp_json = self.Gen_bAB_sp.to_json()
#     open(self.log_path + "gen_bab_sp_model.json", "w").write(Gen_bAB_sp_json)

""" [f0+sp]させる方法と、それの復元方法 """
# va = Voice_Analy()
#
# fs, data = va.get_data_from_wave("materials/data_A\自分の未来は自分の力で手に入れるものです.wav")
# f0, sp, ap = va.get_f0spap(data, fs, fft_size=1024)
#
# f0sp=np.hstack((f0.reshape(-1, 1), sp))
# print(f0sp.shape)
#
# _f0=np.array(f0sp[:, 0])
# print(_f0.shape)
#
# _sp=np.array(f0sp[:, 1:])
# print(_sp.shape)
#
# print(np.mean(sp-_sp))
#
# va.save_wave("res.wav",
#              fs,
#              pw.synthesize(_f0, _sp, ap, fs).astype(np.int16))
# va.save_wave("res1.wav",
#              fs,
#              pw.synthesize(f0, sp, ap, fs).astype(np.int16))

""" stft算出 """
# # データAの波形をSTFT
# stft_a = va.stft(data_a, self.window, self.step)
# # stft_a = stft_a.reshape(stft_a.shape[0], stft_a.shape[1], 1)
#
# # データBのサンプリングレートと波形情報を取得
# sample_rate_b, data_b = va.get_data_from_wave(file_path_b)
# # データBをSTFT
# stft_b = va.stft(data_b, self.window, self.step)
# # stft_b = stft_b.reshape(stft_b.shape[0], stft_b.shape[1], 1)

""" f0, spをもとに学習させるための前処理 """
# データAのサンプリングレートと波形情報を取得
# sample_rate_a, data_a = va.get_data_from_wave(file_path_a)
# data_a = data_a / 32768.0  # 入力音声を-1.0~+1.0に
# f0_a, sp_a, ap_a = va.get_f0spap(data_a, sample_rate_a, fft_size=self.fft_len)
# f0_a = f0_a / 472.0
# sp_a = sp_a / 52.0
# f0sp_a = np.hstack((f0_a.reshape(-1, 1), sp_a))
#
# # データBのサンプリングレートと波形情報を取得
# sample_rate_b, data_b = va.get_data_from_wave(file_path_b)
# data_b = data_b / 32768.0
# f0_b, sp_b, ap_b = va.get_f0spap(data_b, sample_rate_b, fft_size=self.fft_len)
# f0_b = f0_b / 278.0
# sp_b = sp_b / 26.0
# f0sp_b = np.hstack((f0_b.reshape(-1, 1), sp_b))
