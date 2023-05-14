namespace Android;

public partial class Appl : Application
{
	public Appl()
	{
		InitializeComponent();

		MainPage = new AppShell();
	}
}
