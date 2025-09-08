netcdf /home/ana/Desktop/code/ISMIP/processing/✅convert to nc/IsmipF_S3_1.5_0.5-Transient.nc {
  group: results {
    group: TransientSolution {
      dimensions:
        time = 37;
        vertices = 4050;
        Vx_dim2 = 1;
        Vy_dim2 = 1;
        Vz_dim2 = 1;
        Vel_dim2 = 1;
        Pressure_dim2 = 1;
        Thickness_dim2 = 1;
        Surface_dim2 = 1;
        Base_dim2 = 1;
      variables:
        double time(time=37);
          :units = "years";
          :long_name = "time";

        int time_step(time=37);
          :long_name = "time step number";

        double StressbalanceConvergenceNumSteps(time=37, vertices=4050);
          :long_name = "StressbalanceConvergenceNumSteps";

        double Vx(time=37, vertices=4050, Vx_dim2=1);
          :long_name = "Vx";

        double Vy(time=37, vertices=4050, Vy_dim2=1);
          :long_name = "Vy";

        double Vz(time=37, vertices=4050, Vz_dim2=1);
          :long_name = "Vz";

        double Vel(time=37, vertices=4050, Vel_dim2=1);
          :long_name = "Vel";

        double Pressure(time=37, vertices=4050, Pressure_dim2=1);
          :long_name = "Pressure";

        double Thickness(time=37, vertices=4050, Thickness_dim2=1);
          :long_name = "Thickness";

        double Surface(time=37, vertices=4050, Surface_dim2=1);
          :long_name = "Surface";

        double Base(time=37, vertices=4050, Base_dim2=1);
          :long_name = "Base";

    }

  }

  // global attributes:
  :source = "ISSM";
  :original_file = "IsmipF_S3_1.5_0.5-Transient.outbin";
  :creation_date = "2025-09-08T21:16:02";
  :solution_type = "TransientSolution";
  :time_steps = 37L; // long
  :title = "ISSM Simulation Output";
}