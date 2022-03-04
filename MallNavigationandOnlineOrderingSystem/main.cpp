#include <iostream>
#include <vector>
#include <windows.h>
#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
using namespace std;



class login
{
    int x;
    string password;
    string email;
public:
    log()
    {
        system("cls");
        cout<<"FOR SIGN IN PRESS 1"<<endl;
        cout<<"FOR SIGN UP PRESS 2"<<endl;
        cin>>x;

        if(x==1)
        {
            cout<<"ENTER EMAIL"<<endl;
            cin>>email;
            cout<<"ENTER PASSWORD"<<endl;
            cin>>password;
            signin();
        }

        else if(x==2)
        {
            cout<<"ENTER EMAIL"<<endl;
            cin>>email;
            cout<<"ENTER PASSWORD"<<endl;
            cin>>password;
            signup();
        }

        else
        {
            cout<<"WRONG CHOICE"<<endl;
            Sleep(1000);
            log();
        }
    }

    int hashs(string a)
    {
    int b=0,c;
    for(int i=0;i<a.length();i++)
    {
        c=(int(a[i])%5)*2;
        b=10*b+c;
    }
    return b;
    }

    void signup()
    {
        fstream f;
        f.open("login.txt", ios::out | ios::in | ios::app);
        f<<email<<endl;
        f<<hashs(password)<<endl;
        f.close();
        cout<<"SIGNING UP"<<endl;
        Sleep(2000);
        system("cls");
    }

    void signin()
    {
        string line;
        int pass=-1;
        fstream f;
        f.open("login.txt", ios::out | ios::in | ios::app);
        while(getline(f,line))
        {
            if(email.compare(line)==0)
            {
                pass=hashs(password);
                continue;
            }
            if(pass!=-1)
            {
                stringstream ss;
                ss<<pass;

                string s=ss.str();
                if(s.compare(line)==0)
                {
                    cout<<"SIGNING IN!!!"<<endl;
                    Sleep(2000);
                    system("cls");
                    break;
                }
                else
                {
                    cout<<"WRONG PASSWORD"<<endl;
                    Sleep(1000);
                    log();
                }
            }
        }
        if(pass==-1)
        {
            cout<<"WRONG ID"<<endl;
            Sleep(1000);
            log();
        }
        f.close();
    }

};

class nav
{
    int x;
public:
    begins()
    {
        system("cls");
        cout<<endl<<"Choose a category from below"<<endl;
        cout<<"1. FASHION"<<endl;
        cout<<"2. FOOTWEARS"<<endl;
        cout<<"3. ELECTRONICS"<<endl;
        cout<<"4. FOOD"<<endl;
        cout<<"5. PALIKA BAZAAR"<<endl;
        cout<<"6. BIG BAZAAR"<<endl;
        cout<<"7. GAME ZONE"<<endl;
        cout<<"8. EXIT"<<endl;
        cout<<"Enter Index: ";
        cin>>x;
        switch(x)
        {
        case 1:
            fashion();
            Sleep(5000);
            begins();
        case 2:
            footwear();
            Sleep(5000);
            begins();
        case 3:
            electronics();
            Sleep(9000);
            begins();
        case 4:
            food();
            begins();
        case 5:
            palikaBazaar();
            Sleep(3000);
            begins();
        case 6:
            bigBazaar();
            Sleep(3000);
            begins();
        case 7:
            gameZone();
            Sleep(3000);
            begins();
        case 8:
            _Exit(10);
        default:
            cout<<"Please Enter Right Number"<<endl;
            Sleep(900);
            begins();
        }
    }

    void fashion()
    {
        vector<string>fa;
        vector<int>faf;
        vector <int>fas;

        fa.push_back("NIKE");
        fa.push_back("PANTALOONS");
        fa.push_back("LEVIS");
        fa.push_back("BURBERRY");
        fa.push_back("LOUIS VUITTON");
        fas.push_back(27);
        fas.push_back(17);
        fas.push_back(5);
        fas.push_back(7);
        fas.push_back(23);
        faf.push_back(3);
        faf.push_back(2);
        faf.push_back(1);
        faf.push_back(1);
        faf.push_back(3);

        for(int i=0;i<fa.size();i++)
            {
                cout<<fa[i]<<endl<<"  FLOOR: "<<faf[i]<<endl<<"  SHOP: "<<fas[i]<<endl<<endl;
            }
    }

    void footwear()
    {
        vector<string>fo;
        vector<int>fof;
        vector <int>fos;

        fo.push_back("NIKE");
        fo.push_back("WOODLAND");
        fo.push_back("PUMA");
        fo.push_back("ADIDAS");
        fo.push_back("REEBOK");
        fos.push_back(24);
        fos.push_back(18);
        fos.push_back(6);
        fos.push_back(8);
        fos.push_back(26);
        fof.push_back(3);
        fof.push_back(2);
        fof.push_back(1);
        fof.push_back(1);
        fof.push_back(3);

        for(int i=0;i<fo.size();i++)
            {
                cout<<fo[i]<<endl<<"  FLOOR: "<<fof[i]<<endl<<"  SHOP: "<<fos[i]<<endl<<endl;
            }
    }

    void electronics()
    {
        fstream f;
        string line;
        f.open("electronics.txt");
        while(f)
        {
            getline(f,line);
            cout<<line<<endl;
        }
        f.close();
    }


    void food()
    {
        int y;
        cout<<"For Fast Food Enter 1"<<endl;
        cout<<"For Dining Enter 2"<<endl;

        vector<string>f;
        vector<int>ff;
        vector<int>fs;
        //fast food
        f.push_back("FOOD COURT");
        ff.push_back(3);
        fs.push_back(21);
        //restaurants
        f.push_back("Tandoori Village");
        f.push_back("Cafe Coffee Day");
        f.push_back("BBQ");
        f.push_back("Theos");
        f.push_back("Walk In Woods");
        fs.push_back(2);
        fs.push_back(12);
        fs.push_back(22);
        fs.push_back(9);
        fs.push_back(16);
        ff.push_back(1);
        ff.push_back(2);
        ff.push_back(3);
        ff.push_back(1);
        ff.push_back(2);
        foo:
        cin>>y;
        if(y==1)
        {
            cout<<f[0]<<"       FLOOR: "<<ff[0]<<"       SHOP: "<<fs[0]<<endl<<endl;
            Sleep(3000);
        }
        else if(y==2)
        {
            for(int i=1;i<f.size();i++)
            {
                cout<<f[i]<<endl<<"  FLOOR: "<<ff[i]<<endl<<"  SHOP: "<<fs[i]<<endl<<endl;
            }
            Sleep(5000);
        }
        else
        {
            cout<<"Please Enter 1 or 2"<<endl;
            goto foo;
        }
    }

    void palikaBazaar()
    {
        cout<<"FLOOR: BASEMENT       SHOP: 1 "<<endl<<endl;
    }
    void bigBazaar()
    {
        cout<<"FLOOR: 1,2       SHOP: 4,14 "<<endl<<endl;
    }
    void gameZone()
    {
        cout<<"FLOOR: 2       SHOP: 20 "<<endl<<endl;
    }
};



//..............................................................................................................................................................................

/*
int billamt=0;

class bill
{
	public:
		bill(int type);

};
*/
//.......................................................................................................................................................................
class cloth
{

	struct clothlist
	{
		string name;
		string colour;
	//	int size;
		int price;
	//	string i;
	};

	public:
	int size_of_clothlist;
	//int tax1;
	//int tax2;
	cloth()
	{
		//tax1=5;
		//tax2=12;
		size_of_clothlist=sizeof(clothlist);
	}


};


class partywearBoys:virtual public cloth
{
	public:
		struct clothlist c1[5]={ { "Three-Piece Suit" , "Wine       " , 10000 } , { "Kurta Pajama    " ,"Red & White" , 6000 } , { "Kurta           " ,"Black      " , 2500  } , { "Blazer          " ,"Blue       " , 3000  } , { "Formal Shirt    " ,"White Black" , 2000  }   };
		int displayParty(int h)
		{
			int i;
			cout<<"\nItem\tColour\tPrice"<<endl;
			for(i=0;i<5;i++)
			{
				cout<<i+h+1<<". "<<c1[i].name<<"\t";
				if(c1[i].name.length()<8)
				cout<<"        ";
				cout<<c1[i].colour<<"\t";
				if(c1[i].colour.length()<8)
				cout<<"        ";
				cout<<c1[i].price<<endl;
			}
			return i+h;
		}


};

class casualwearBoys:virtual public cloth
{
	public:
		struct clothlist c2[5]={ { "Shirt       " , "Blue " , 500 } , { "Rugged Jeans" ,"Blue " , 600 } , { "SweatShirt  " ,"Black" , 1000  } , { "Pant        " ,"Blue " , 300  } , { "T-Shirt     " ,"White" , 200  }   };
		int displayCasual(int h)
		{
			int i;
			cout<<"\nItem\t        Colour\t        Price"<<endl;
			for(i=0;i<5;i++)
			{
				cout<<i+h+1<<". "<<c2[i].name<<"\t";
				if(c2[i].name.length()<8)
				cout<<"        ";
				cout<<c2[i].colour<<"\t";
				if(c2[i].colour.length()<=8)
				cout<<"        ";
				cout<<c2[i].price<<endl;
			}
			return i+h;
		}


};

class gharBoys:virtual public cloth
{
	public:
		struct clothlist c3[2]={ { "Pyjamas   " , "Black" , 500 } , { "TrackSuit" ,"Blue" , 1000 }};
		int displayghar(int h)
		{
			int i;
			cout<<"\nItem\t        Colour\t   Price"<<endl;
			for(i=0;i<2;i++)
			{
				cout<<i+h+1<<". "<<c3[i].name<<"\t";
				if(c3[i].name.length()<8)
				cout<<"        ";
				cout<<c3[i].colour;
				if(c3[i].colour.length()<8)
				cout<<"        ";
				cout<<c3[i].price<<endl;
			}
			return i+h;
		}


};
class Mittal:public casualwearBoys,public partywearBoys,public gharBoys
{
	public:
		void display()
		{
			int h=0;
		//	cout<<"\nItem\t        Colour\t        Price"<<endl;
			h=displayParty(h);
			h=displayCasual(h);
			h=displayghar(h);

		}

};


//.......................................................................................................................................................................

class food
{
	struct foodlist
	{
		string namefood;
		int cost;
	};


	public:

		int tax;
		int h;
		int size_of_foodlist;
		food()
		{
			tax=18;
			size_of_foodlist=sizeof(foodlist);
		}


	//	foodlist *a;
	//	virtual void display()=0;
};

class Mburger:virtual public food
{

	public:
		struct foodlist a[5]={ {"McBurger    ",75} , {"McPuff      ",40} , {"McVeggie    ",60} , {"McChicken   ",100} , {"McWrap      ",60} };

		int displayB(int h)
		{
			int i;
			cout<<"\nFoodName\t"<<"Price"<<endl;
			for(i=0;i<5;i++)
			{
				cout<<i+h+1<<". "<<a[i].namefood<<"\t";
				if(a[i].namefood.length()<12)
				cout<<"        ";
				cout<<a[i].cost<<endl;
			}
			return i+h;
		}

};

class Mdrinks:virtual public food
{
	public:
		struct foodlist aa[4]={ {"Soda        ",50} , {"Smoothie    ",75} , {"Pepsi       ",60} , {"Coca Cola   ",60} };

		int displayd(int h)
		{
			int i;
			cout<<"\nFoodName\t"<<"Price"<<endl;
			for(i=0;i<4;i++)
			{
				cout<<i+h+1<<". "<<aa[i].namefood<<"\t";
				if(aa[i].namefood.length()<12)
				cout<<"        ";
				cout<<aa[i].cost<<endl;
			}
			return i+h;
		}

};

class McDessert:virtual public food
{
	public:
		struct foodlist aaa[5]={ {"McBrownie   ",150} , {"McVanilla   ",75} , {"McFlurry    ",80} , {"McChoco     ",60} , {"McStrawberry",85} };

		int displayde(int h)
		{
			int i;
			cout<<"\nFoodName    \t"<<"Price"<<endl;
			for(i=0;i<5;i++)
			{
				cout<<i+h+1<<". "<<aaa[i].namefood<<"\t";
/*				if(aaa[i].namefood.length()<12)
				cout<<"        ";                         */
				cout<<aaa[i].cost<<endl;
			}
			return i+h;
		}

};

class McDonald: public Mburger ,public Mdrinks,public McDessert
{
/*	struct Bill{
		string a1;
		int a3;
	//	string a2;
		struct Bill *next;
		int a4;
	}*q;
	int amount;

	struct Bill *head;
*/
	public:
		McDonald()
		{
			//amount=0;
			//q=new struct Bill;

		}
		void display()
		{
			int h=0;
			h=displayB(h);
			h=displayd(h);
			h=displayde(h);
		}
		//void calBill(int h)






};




//......................................................................................................................................................................................


class foot
{
public:
    struct footlist
    {
        string fname;
        int fprice;
    };

    int size_of_footlist;
    foot()
    {
        size_of_footlist=sizeof(footlist);
    }
};

class SportsShoes: virtual public foot
{
    public:
        struct footlist z[3]={ {"Sparx Sports Shoes ",1050}, {"Adidas Sports Shoes",3299}, {"Puma Sports Shoes  ",2799} };

        int displayf(int h)
        {
        	int i;
            cout<<"\nName of the Product\t"<<"Price"<<endl;
            for (i=0;i<3;i++)
            {
                cout<<i+h+1<<". "<<z[i].fname<<"\t";
                if(z[i].fname.length()<12)
                    cout<<"               ";
                cout<<z[i].fprice<<endl;
            }

        return i+h;
		}

};

class Ladies : virtual public foot
{
public:
    struct footlist zz[5]= { {"High heels",2499}, {"Low heels ",1999}, {"Sandal    ",1499}, {"Jooti     ",1299}, {"Sneakers  ",1999} };

    int displayff(int h)
        {
        	int i;
            cout<<"\nName of the Product\t"<<"Price"<<endl;
            for (i=0;i<5;i++)
            {
                cout<<i+h+1<<". "<<zz[i].fname<<"\t";
                if(zz[i].fname.length()<12)
                    cout<<"            ";
                cout<<zz[i].fprice<<endl;
            }
            return i+h;
        }
};

class slippers: virtual public foot
{
public:
    struct footlist zzz[3]= { {"Flip flops",999}, {"Sliders   ",1299}, {"Crocs     ",1899} };

    int displayfff(int h)
        {
        	int i;
            cout<<"\nName of the Product\t"<<"Price"<<endl;
            for (i=0;i<3;i++)
            {
                cout<<i+h+1<<". "<<zzz[i].fname<<"\t";
                if(zzz[i].fname.length()<12)
                    cout<<"            ";
                cout<<zzz[i].fprice<<endl;
            }
            return i+h;
        }

};


class ShoeShop:public SportsShoes, public Ladies, public slippers
{
	public:
	void display()
	{
		int h=0;
		h=displayf(h);
    	h=displayff(h);
    	h=displayfff(h);

	}
};


//......................................................................................................................................................................................

class billingSystem
{
	struct Bill{
		string a1;
		int a3;
		string a2;
		struct Bill *next;
		int a4;
	}*q;
	int amount;

	struct Bill *head;
	public:
		billingSystem()
		{
			q=new struct Bill;
			head=q;
			q->next=NULL;
		//	cout<<sizeof(q);
			amount=0;
		}
		void addBill(string a,string b,int c,int d)
		{
			amount=amount+(c*d);
			int jh=1;
			struct Bill *aa;
			aa=head;
			while(aa)
			{

				if(a==aa->a1)
				{
					aa->a4+=d;
					jh=0;
					break;

				}
				aa=aa->next;
			}
			if(jh)
			{
				q->a1=a;
				q->a2=b;
				q->a3=c;
				q->a4=d;
				q->next=new struct Bill;
				q=q->next;
				q->next=NULL;
			}
		}

		void addBill(string a,int c,int d)
		{
			amount=amount+(c*d);
			int jh=1;
			struct Bill *aa;
			aa=head;
			while(aa)
			{

				if(a==aa->a1)
				{
					aa->a4+=d;
					jh=0;
					break;

				}
				aa=aa->next;
			}
			if(jh)
			{
				q->a1=a;
			//	q->a2=b;
				q->a3=c;
				q->a4=d;
				q->next=new struct Bill;
				q=q->next;
				q->next=NULL;
			}
		}
		void print()
		{
			struct Bill *p;
			p=head;
			while(p->next)
			{
				cout<<endl<<p->a1<<"\t"<<p->a2<<"\t"<<p->a3<<"\t"<<p->a4<<endl;

				p=p->next;

			}
			cout<<"\nYour Total Bill is "<<amount<<endl;
		}

		void printabc()
		{
			struct Bill *p;
			p=head;
			while(p->next)
			{
				cout<<endl<<p->a1<<"\t"<<p->a3<<"\t"<<p->a4<<endl;

				p=p->next;

			}
			cout<<"\nYour Total Bill is "<<amount<<endl;
		}

};






//......................................................................................................................................................................................
int main()
{
	  login l;
	  l.log();

	cout<<"WELOCME TO GAUR SHOPPING COMPLEX\n";

	cout<<"Would you like to EXPLORE the Mall or ORDER ONLINE (Press 1 to explore Press 2 to order online)";

	cout<<endl;
	int press;
	cin>>press;
	system("cls");
	if(press==1)
	{//adrees
		nav n;
    	n.begins();

	}
	else if(press ==2)
	{

		cout<<"What would you like to order?\n1. Food \n2. Clothes\n3. Footwear (Enter the corresponding number)"<<endl;
		int choose;
		cin>>choose;
		system("cls");
		billingSystem b1;
		if(choose== 1)
		{
			McDonald d;
			//cout<<endl<<sizeof(d.aa)/d.size_of_foodlist;
			string yn;
//			cin>>yn;
			yn="y";
			while(yn=="y" or yn=="Y")
			{
				d.display();
				int nf;
				cout<<"Enter the number of food items you want to order"<<endl;
				cin>>nf;
				for(int i=0;i<nf;i++)
				{
					cout<<endl<<"Enter the index no. of Food item you want to order"<<endl;
					int noo;
					cin>>noo;
					cout<<endl<<"Enter the quantity\n";
					int quan;
					cin>>quan;
					int h1=(sizeof(d.a)/d.size_of_foodlist);
					int h2=h1+(sizeof(d.aa)/d.size_of_foodlist);
					int h3 =h2+(sizeof(d.aaa)/d.size_of_foodlist);
					if(noo<=h1)
					{
						b1.addBill(d.a[noo-1].namefood,d.a[noo-1].cost,quan);

					}
					else if(noo<=h2)
					{
						b1.addBill(d.aa[noo-1-h1].namefood,d.aa[noo-1-h1].cost,quan);
					}
					else if(noo<=h3)
					{
						b1.addBill(d.aaa[noo-1-h2].namefood,d.aaa[noo-1-h2].cost,quan);
					}
				}
				cout<<"Do you want to order more??(y for yes and n for no)\n\n";
				cin>>yn;
				system("cls");
			}
			b1.printabc();

		}
		else if(choose == 2)
		{
			Mittal m;
			//cout<<endl<<sizeof(d.aa)/d.size_of_foodlist;
			string yn;
			//cin>>yn;
			yn="y";
			while(yn=="y" or yn=="Y")
			{
				m.display();
				int nf;
				cout<<"Enter the number of clothe items you want to order\n\n";
				cin>>nf;
				for(int i=0;i<nf;i++)
				{
				cout<<endl<<"Enter the index no. of clothe item you want to order"<<endl;
				int noo;
				cin>>noo;
				cout<<endl<<"Enter the quantity\n";
				int quan;
				cin>>quan;
				int h1=(sizeof(m.c1)/m.size_of_clothlist);
				int h2=h1+(sizeof(m.c2)/m.size_of_clothlist);
				int h3 =h2+(sizeof(m.c3)/m.size_of_clothlist);
				if(noo<=h1)
				{
					b1.addBill(m.c1[noo-1].name,m.c1[noo-1].colour,m.c1[noo-1].price,quan);

				}
				else if(noo<=h2)
				{
					b1.addBill(m.c2[noo-1-h1].name,m.c2[noo-1-h1].colour,m.c2[noo-h1-1].price,quan);
				}
				else if(noo<=h3)
				{
						b1.addBill(m.c3[noo-1-h2].name,m.c3[noo-1-h2].colour,m.c3[noo-1-h2].price,quan);
				}
			}
				cout<<"Do you want to order more??(y for yes and n for no)\n\n";
				cin>>yn;
				system("cls");
			}

			b1.print();

		}
		else if(choose == 3)
		{
			ShoeShop s;
            string yn;
			//cin>>yn;
			yn="y";
			while(yn=="y" or yn=="Y")
			{

				s.display();
				int nf;
				cout<<"Enter the number of clothe items you want to order\n\n";
				cin>>nf;
				for(int i=0;i<nf;i++)
				{
					cout<<endl<<"Enter the index no. of Shoe item you want to order"<<endl;
					int noo;
					cin>>noo;
					cout<<endl<<"Enter the quantity\n";
					int quan;
					cin>>quan;
					int h11=(sizeof(s.z)/s.size_of_footlist);
					int h21=h11+(sizeof(s.zz)/s.size_of_footlist);
					int h31 =h21+(sizeof(s.zzz)/s.size_of_footlist);
					if(noo<=h11)
					{
						b1.addBill(s.z[noo-1].fname,s.z[noo-1].fprice,quan);

					}
					else if(noo<=h21)
					{
						b1.addBill(s.zz[noo-1-h11].fname,s.zz[noo-1-h11].fprice,quan);
					}
					else if(noo<=h31)
					{
						b1.addBill(s.zzz[noo-1-h21].fname,s.zzz[noo-1-h21].fprice,quan);
					}
				}
				cout<<"Do you want to order More??(y for yes and n for no)";
				cin>>yn;
				system("cls");
			}
			b1.printabc();
		}
		//system("cls");
	}
}
