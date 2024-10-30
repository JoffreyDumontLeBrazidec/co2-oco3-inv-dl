#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of pres_df_explo
# TODO: 
#
#

print(df_pos[df_pos["pred_success"]==True].mean())
print(df_pos[df_pos["pred_success"]==False].mean())

df_group = df_pos.groupby('pred_success').mean().reset_index()
print (df_group)

df_pos['ext price'].describe()
df_test.loc[(df_test.positivity=="positive") & (df_test.pred_success==False)].describe()
